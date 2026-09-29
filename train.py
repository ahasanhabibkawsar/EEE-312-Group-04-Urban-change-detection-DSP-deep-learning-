"""
Training script for Urban Change Detection (v2).

Examples
--------
    # Main model (RGB + Canny, 4 channels)
    python train.py

    # DSP ablation: identical settings, RGB only (3 channels)
    python train.py --no-canny

    # Ablation: no auxiliary edge loss
    python train.py --edge-weight 0 --out-dir checkpoints/ablation_no_edge

    # Resume an interrupted run
    python train.py --resume

    # 2-minute smoke test of the whole pipeline
    python train.py --epochs 1 --max-train-batches 20 --max-val-batches 4 --out-dir checkpoints/smoke

Outputs (in --out-dir, default checkpoints/ or checkpoints/ablation_rgb_only/)
    best_model.pth          best validation F1 (dataset-level)
    latest_model.pth        last epoch, with optimizer state (for --resume)
    best_threshold.json     threshold chosen on the VALIDATION set
    training_history.csv    per-epoch losses and validation metrics
"""

import argparse
import csv
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import config
from dataset import create_dataloaders
from losses import MultiTaskChangeLoss
from models import SiameseUNet, save_model_checkpoint
from training import Trainer, warmup_cosine_lambda


HISTORY_FIELDS = [
    "epoch", "time_s", "lr",
    "train_loss", "train_bce", "train_dice", "train_edge",
    "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1", "val_iou", "val_kappa",
    "val_best_threshold", "val_best_f1", "val_best_iou",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the v2 Siamese U-Net on LEVIR-CD")

    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--crops-per-image", type=int, default=config.TRAIN_CROPS_PER_IMAGE)
    parser.add_argument("--edge-weight", type=float, default=config.EDGE_LOSS_WEIGHT)
    parser.add_argument("--no-canny", action="store_true",
                        help="RGB-only ablation (3-channel input, no DSP edge channel)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Random encoder initialisation (no ImageNet weights)")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--device", type=str, default=None, help="mps | cuda | cpu")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    return parser.parse_args()


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_history(path, history):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def main():
    args = parse_args()
    set_seed(args.seed)

    use_canny = not args.no_canny
    in_channels = 4 if use_canny else 3

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = config.CHECKPOINT_DIR if use_canny else config.CHECKPOINT_DIR / "ablation_rgb_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / "best_model.pth"
    latest_path = out_dir / "latest_model.pth"
    history_path = out_dir / "training_history.csv"
    threshold_path = out_dir / "best_threshold.json"

    device = get_device(args.device)

    print("=" * 70)
    print("URBAN CHANGE DETECTION - TRAINING (v2)")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Input           : {'RGB + Canny (4 ch)' if use_canny else 'RGB only (3 ch)'}")
    print(f"Crop size       : {config.IMAGE_SIZE}  x {args.crops_per_image} crops / image / epoch")
    print(f"Batch size      : {args.batch_size}")
    print(f"Epochs          : {args.epochs}")
    print(f"Edge loss weight: {args.edge_weight}")
    print(f"Output          : {out_dir}")

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_canny=use_canny,
        crops_per_image=args.crops_per_image,
        seed=args.seed,
    )

    print(f"Train           : {len(train_loader.dataset.file_names)} images -> "
          f"{len(train_loader)} batches / epoch")
    print(f"Validation      : {len(val_loader.dataset)} full-resolution images")

    # -----------------------------------------------------
    # Model, loss, optimiser, schedule
    # -----------------------------------------------------

    model = SiameseUNet(in_channels=in_channels, pretrained=not args.no_pretrained).to(device)
    print(f"Parameters      : {sum(p.numel() for p in model.parameters()):,}")

    criterion = MultiTaskChangeLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT,
        edge_weight=args.edge_weight,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader) if args.max_train_batches is None \
        else min(len(train_loader), args.max_train_batches)
    scheduler = LambdaLR(
        optimizer,
        warmup_cosine_lambda(
            total_steps=args.epochs * steps_per_epoch,
            warmup_steps=config.WARMUP_EPOCHS * steps_per_epoch,
        ),
    )

    trainer = Trainer(model, criterion, optimizer, device,
                      scheduler=scheduler, grad_clip=config.GRAD_CLIP_NORM)

    # -----------------------------------------------------
    # Resume
    # -----------------------------------------------------

    start_epoch = 1
    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    if args.resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_f1 = state.get("best_f1", -1.0)
        epochs_without_improvement = state.get("epochs_without_improvement", 0)
        history = state.get("history", [])
        print(f"Resumed from epoch {state['epoch']} (best val F1 {best_f1:.4f})")

    # -----------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------

    threshold = config.DEFAULT_THRESHOLD

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}  (lr {optimizer.param_groups[0]['lr']:.2e})")

        train_stats = trainer.train_one_epoch(train_loader, max_batches=args.max_train_batches)
        val = trainer.validate(val_loader, threshold=config.DEFAULT_THRESHOLD,
                               max_batches=args.max_val_batches)

        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "time_s": round(elapsed, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "train_bce": train_stats["bce"],
            "train_dice": train_stats["dice"],
            "train_edge": train_stats["edge"],
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_f1": val["f1"],
            "val_iou": val["iou"],
            "val_kappa": val["kappa"],
            "val_best_threshold": val["best_threshold"],
            "val_best_f1": val["best_f1"],
            "val_best_iou": val["best_iou"],
        }
        history.append(row)
        write_history(history_path, history)

        print(f"  train loss {train_stats['loss']:.4f} "
              f"(bce {train_stats['bce']:.4f}, dice {train_stats['dice']:.4f}, edge {train_stats['edge']:.4f})")
        print(f"  val   loss {val['loss']:.4f} | F1@0.50 {val['f1']:.4f} IoU {val['iou']:.4f} | "
              f"best F1 {val['best_f1']:.4f} @ t={val['best_threshold']:.2f} | {elapsed / 60:.1f} min")

        metrics_to_save = {k: v for k, v in val.items() if k != "sweep"}

        if val["best_f1"] > best_f1:
            best_f1 = val["best_f1"]
            threshold = val["best_threshold"]
            epochs_without_improvement = 0

            save_model_checkpoint(
                best_path, model, epoch, metrics_to_save, threshold, use_canny,
            )
            config.save_best_threshold(threshold, f1=best_f1, path=threshold_path)
            print(f"  ✔ new best model saved (val F1 {best_f1:.4f}, threshold {threshold:.2f})")
        else:
            epochs_without_improvement += 1

        save_model_checkpoint(
            latest_path, model, epoch, metrics_to_save, threshold, use_canny,
            optimizer=optimizer, scheduler=scheduler,
            extra={
                "best_f1": best_f1,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            },
        )

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    # The main run also refreshes the project-level history file used by the reports
    if out_dir.resolve() == config.CHECKPOINT_DIR.resolve():
        shutil.copyfile(history_path, config.PROJECT_ROOT / "training_history.csv")

    print("\n" + "=" * 70)
    print(f"Training complete. Best validation F1 = {best_f1:.4f} at threshold {threshold:.2f}")
    print(f"Best model : {best_path}")
    print("Next step  : python evaluate.py" + ("" if use_canny else f" --checkpoint {best_path}"))
    print("=" * 70)


if __name__ == "__main__":
    main()
