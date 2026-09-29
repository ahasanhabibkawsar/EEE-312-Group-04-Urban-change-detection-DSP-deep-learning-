"""
============================================================
SANITY CHECK (v2)  -  run this before a long training run
============================================================

    python sanity_check.py

Takes 1-3 minutes. Verifies every fixed component on real LEVIR-CD
data and prints PASS / FAIL for each test:

  1. Data pipeline   : shapes, channel standardisation, binary masks
  2. Model shapes    : 256 crops (train) and 1024 images (sliding window)
  3. BAM batch fix   : a sample's prediction must NOT depend on the
                       other samples in its batch (the v1 bug)
  4. Sliding window  : for a 256 x 256 input it equals the direct pass
  5. Loss + backward : finite multi-task loss, gradients reach all parts
  6. Overfitting     : loss must drop on a single fixed batch
  7. Metrics         : ThresholdSweep == direct counting; IoU = F1/(2-F1)
  8. Checkpoint      : save -> load round trip gives identical output
  9. Speed           : seconds per training batch -> epoch time estimate
============================================================
"""

import importlib.util
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import config
from dataset import create_train_dataset, create_validation_dataset
from losses import MultiTaskChangeLoss
from models import SiameseUNet, save_model_checkpoint, build_model_from_checkpoint
from training.metrics import ThresholdSweep, calculate_confusion_matrix, metrics_from_counts


RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = get_device()
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # 1. Data pipeline
    # ------------------------------------------------------------------
    train_ds = create_train_dataset(crops_per_image=1)
    val_ds = create_validation_dataset()

    sample = train_ds[0]
    x = sample["before"]
    check("train crop shape", tuple(x.shape) == (4, config.IMAGE_SIZE, config.IMAGE_SIZE), str(tuple(x.shape)))

    stack = torch.stack([train_ds[i]["before"] for i in range(min(16, len(train_ds)))])
    means = stack.mean(dim=(0, 2, 3))
    stds = stack.std(dim=(0, 2, 3))
    print(f"       channel means {means.numpy().round(2)}  stds {stds.numpy().round(2)}")
    check("channels standardised (|mean| < 1, 0.3 < std < 2)",
          bool((means.abs() < 1).all() and (stds > 0.3).all() and (stds < 2).all()))
    check("edge channel on same scale as RGB",
          0.3 < float(stds[3] / stds[:3].mean()) < 3.0,
          f"edge std / RGB std = {float(stds[3] / stds[:3].mean()):.2f} (v1: ~0.002)")

    full = val_ds[0]
    check("validation image at native resolution", full["before"].shape[-1] == 1024, str(tuple(full["before"].shape)))
    check("mask is binary", set(torch.unique(full["mask"]).tolist()) <= {0.0, 1.0})

    # ------------------------------------------------------------------
    # 2. Model shapes
    # ------------------------------------------------------------------
    model = SiameseUNet(in_channels=4, pretrained=False).to(device)
    print(f"       parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    b = torch.randn(2, 4, 256, 256, device=device)
    a = torch.randn(2, 4, 256, 256, device=device)
    change, edge = model(b, a, return_edges=True)
    check("train forward 256 -> full-resolution logits",
          tuple(change.shape) == (2, 1, 256, 256) and tuple(edge.shape) == (2, 1, 256, 256))

    model.eval()
    with torch.no_grad():
        big = model(full["before"][None].to(device), full["after"][None].to(device))
    check("eval forward 1024 x 1024 (sliding window)", tuple(big.shape) == (1, 1, 1024, 1024))

    with torch.no_grad():
        odd = model(torch.randn(1, 4, 300, 413, device=device), torch.randn(1, 4, 300, 413, device=device))
    check("arbitrary image size (300 x 413)", tuple(odd.shape) == (1, 1, 300, 413))

    # ------------------------------------------------------------------
    # 3. BAM batch independence (v1 bug)
    # ------------------------------------------------------------------
    # Make BAM active (gamma is 0 at initialisation)
    with torch.no_grad():
        for bam in model.fusion.bams.values():
            bam.gamma.fill_(1.0)

    crops = [train_ds[i] for i in range(4)]
    B = torch.stack([c["before"] for c in crops]).to(device)
    A = torch.stack([c["after"] for c in crops]).to(device)

    with torch.no_grad():
        batched = model(B, A)
        single = torch.cat([model(B[i:i + 1], A[i:i + 1]) for i in range(4)])
    diff = float((batched - single).abs().max())
    check("v2 BAM: prediction independent of batch composition", diff < 1e-3, f"max diff {diff:.2e}")

    v1_fusion = Path(__file__).parent / "_v1_baseline" / "models" / "fusion.py"
    if v1_fusion.exists():
        spec = importlib.util.spec_from_file_location("v1_fusion", v1_fusion)
        v1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v1)
        old_bam = v1.BAM(64).eval()
        fb, fa = torch.randn(4, 64, 16, 16), torch.randn(4, 64, 16, 16)
        with torch.no_grad():
            out_batch = old_bam(fb, fa)[0]
            out_single = torch.cat([old_bam(fb[i:i + 1], fa[i:i + 1])[0] for i in range(4)])
        old_diff = float((out_batch - out_single).abs().max())
        print(f"       (for the report) v1 BAM batch-vs-single difference: {old_diff:.3f}  <- the bug")

    with torch.no_grad():
        for bam in model.fusion.bams.values():
            bam.gamma.fill_(0.0)

    # ------------------------------------------------------------------
    # 4. Sliding window == direct pass on one tile
    # ------------------------------------------------------------------
    with torch.no_grad():
        direct = model._forward_padded(B[:1], A[:1])
        tiled = model.sliding_window_forward(B[:1], A[:1])
    sw = float((direct - tiled).abs().max())
    check("sliding window == direct on a 256 tile", sw < 1e-4, f"max diff {sw:.2e}")

    # ------------------------------------------------------------------
    # 5. Loss + gradients
    # ------------------------------------------------------------------
    model.train()
    criterion = MultiTaskChangeLoss(edge_weight=config.EDGE_LOSS_WEIGHT)
    M = torch.stack([c["mask"] for c in crops]).to(device)

    model.zero_grad()
    change, edge = model(B, A, return_edges=True)
    loss, parts = criterion(change, M, edge)
    loss.backward()
    check("multi-task loss finite", torch.isfinite(loss).item(), str({k: round(v, 4) for k, v in parts.items()}))

    def grad_norm(module):
        return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)

    check("gradients reach encoder / fusion / decoder / edge head",
          all(grad_norm(m) > 0 for m in (model.encoder, model.fusion, model.decoder, model.edge_head)))

    # ------------------------------------------------------------------
    # 6. Overfit one batch
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(25):
        optimizer.zero_grad()
        change, edge = model(B, A, return_edges=True)
        loss, _ = criterion(change, M, edge)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    check("loss decreases when overfitting one batch", losses[-1] < 0.7 * losses[0],
          f"{losses[0]:.3f} -> {losses[-1]:.3f}")

    # ------------------------------------------------------------------
    # 7. Metrics
    # ------------------------------------------------------------------
    probs = torch.rand(2, 1, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.8).float()
    sweep = ThresholdSweep()
    sweep.update(probs, target)
    direct_counts = calculate_confusion_matrix(probs >= 0.4, target)
    check("ThresholdSweep counts == direct counts", sweep.counts_at(0.4) == direct_counts)
    m = metrics_from_counts(*direct_counts)
    check("dataset-level IoU = F1 / (2 - F1)", abs(m["iou"] - m["f1"] / (2 - m["f1"])) < 1e-9)

    # ------------------------------------------------------------------
    # 8. Checkpoint round trip
    # ------------------------------------------------------------------
    model.eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pth"
        save_model_checkpoint(path, model, 0, {"f1": 0.0}, 0.5, use_canny=True)
        loaded, _ = build_model_from_checkpoint(path, device)
        with torch.no_grad():
            d = float((model(B[:1], A[:1]) - loaded(B[:1], A[:1])).abs().max())
    check("checkpoint save / load round trip", d < 1e-5, f"max diff {d:.2e}")

    # ------------------------------------------------------------------
    # 9. Speed
    # ------------------------------------------------------------------
    model = SiameseUNet(in_channels=4, pretrained=False).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    bs = config.BATCH_SIZE
    b = torch.randn(bs, 4, 256, 256, device=device)
    a = torch.randn(bs, 4, 256, 256, device=device)
    y = (torch.rand(bs, 1, 256, 256, device=device) > 0.9).float()

    for step in range(8):
        if step == 3:
            if device.type == "mps":
                torch.mps.synchronize()
            start = time.time()
        optimizer.zero_grad()
        change, edge = model(b, a, return_edges=True)
        loss, _ = criterion(change, y, edge)
        loss.backward()
        optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    per_batch = (time.time() - start) / 5

    batches = (445 * config.TRAIN_CROPS_PER_IMAGE) // bs
    print(f"\n       {per_batch:.2f} s / training batch (batch {bs})")
    print(f"       ≈ {per_batch * batches / 60:.0f} min training + validation per epoch "
          f"→ roughly {per_batch * batches * config.NUM_EPOCHS * 1.3 / 3600:.1f} h for {config.NUM_EPOCHS} epochs")

    # ------------------------------------------------------------------
    passed = sum(ok for _, ok in RESULTS)
    print("\n" + "=" * 60)
    print(f"{passed}/{len(RESULTS)} checks passed")
    print("=" * 60)
    if passed == len(RESULTS):
        print("All good - start training with:  python train.py")


if __name__ == "__main__":
    main()
