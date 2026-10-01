"""
============================================================
URBAN CHANGE DETECTION - EVALUATION (v2)
============================================================

Full-resolution evaluation of the trained model.

Every 1024 x 1024 test image is predicted with the overlapping sliding
window (native 0.5 m / pixel, no down-sampling). Metrics are reported
two ways:

    dataset-level (micro)  TP/FP/FN summed over all test pixels.
                           This is the number to report / compare
                           with LEVIR-CD papers.
    per-image (macro)      mean of per-image scores (for analysis only).

The decision threshold is the one selected on the VALIDATION set during
training (checkpoints/best_threshold.json). The test set is never used
to choose anything.

Examples
--------
    python evaluate.py                     # test set, validation threshold
    python evaluate.py --tta               # + test-time augmentation
    python evaluate.py --tune              # re-tune threshold on val, then test
    python evaluate.py --split val
    python evaluate.py --checkpoint checkpoints/ablation_rgb_only/best_model.pth \
                       --out-dir evaluation_results_rgb_only
============================================================
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import create_validation_dataset, create_test_dataset
from models import build_model_from_checkpoint, checkpoint_threshold, predict_probabilities
from preprocessing.dsp_processing import denormalize_for_display
from training.metrics import ThresholdSweep, metrics_from_counts, calculate_confusion_matrix
from evaluation.visualization import (
    save_confusion_matrix,
    save_metric_summary,
    save_prediction_panel,
    save_threshold_curve,
)
from evaluation.report import save_text_report, save_json_report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the change-detection model")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_PATH))
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the validation-selected threshold")
    parser.add_argument("--tune", action="store_true",
                        help="Select the threshold on the validation set first")
    parser.add_argument("--tta", action="store_true", help="4-view test-time augmentation")
    parser.add_argument("--overlap", type=int, default=config.INFERENCE_TILE_OVERLAP)
    parser.add_argument("--num-vis", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=str(config.PROJECT_ROOT / "evaluation_results"))
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_threshold(args, checkpoint, checkpoint_path):
    """Explicit --threshold > best_threshold.json next to checkpoint > checkpoint > default."""

    if args.threshold is not None:
        return args.threshold, "command line"

    json_path = Path(checkpoint_path).parent / "best_threshold.json"
    if json_path.exists():
        return config.load_best_threshold(json_path), f"validation ({json_path.name})"

    if "threshold" in checkpoint:
        return checkpoint_threshold(checkpoint), "validation (checkpoint)"

    return config.DEFAULT_THRESHOLD, "default"


@torch.no_grad()
def run_split(model, dataset, device, threshold, tta, num_vis, out_dir, label):
    """
    Predict every image of `dataset`; return sweep + per-image rows.
    """

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    sweep = ThresholdSweep()
    rows = []
    vis_dir = out_dir / "predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    for index, batch in enumerate(loader):
        before = batch["before"].to(device)
        after = batch["after"].to(device)
        mask = batch["mask"]
        filename = batch["filename"][0]

        probability = predict_probabilities(model, before, after, tta=tta).cpu()

        sweep.update(probability, mask)

        counts = calculate_confusion_matrix(probability >= threshold, mask)
        metrics = metrics_from_counts(*counts)
        rows.append({
            "filename": filename,
            **{k: metrics[k] for k in ("precision", "recall", "f1", "iou", "accuracy",
                                       "tp", "tn", "fp", "fn")},
            "gt_change_percent": 100.0 * float(mask.mean()),
            "pred_change_percent": 100.0 * float((probability >= threshold).float().mean()),
        })

        if index < num_vis:
            save_prediction_panel(
                denormalize_for_display(batch["before"][0]),
                denormalize_for_display(batch["after"][0]),
                mask[0, 0].numpy(),
                probability[0, 0].numpy(),
                threshold,
                vis_dir / f"{index + 1:03d}_{Path(filename).stem}.png",
                title=f"{label}: {filename}",
                metrics=metrics,
            )

        if (index + 1) % 16 == 0 or index + 1 == len(dataset):
            print(f"  {index + 1:3d}/{len(dataset)} images  ({time.time() - start:.0f}s)", flush=True)

    return sweep, rows


def main():
    args = parse_args()
    device = get_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("URBAN CHANGE DETECTION - MODEL EVALUATION (v2)")
    print("=" * 60)

    model, checkpoint = build_model_from_checkpoint(args.checkpoint, device)
    model.tile_overlap = args.overlap
    use_canny = model.in_channels == 4

    print(f"Checkpoint : {args.checkpoint}  (epoch {checkpoint.get('epoch', '?')})")
    print(f"Input      : {'RGB + Canny' if use_canny else 'RGB only'}")
    print(f"Device     : {device}   TTA: {args.tta}   tile overlap: {args.overlap}px")

    # -----------------------------------------------------
    # Threshold (validation only)
    # -----------------------------------------------------

    if args.tune:
        print("\nTuning threshold on the VALIDATION set ...")
        val_sweep, _ = run_split(model, create_validation_dataset(use_canny), device,
                                 config.DEFAULT_THRESHOLD, args.tta, 0, out_dir / "val_tuning", "val")
        best = val_sweep.best("f1")
        threshold, source = best["threshold"], "validation (--tune)"
        config.save_best_threshold(threshold, f1=best["f1"],
                                   path=Path(args.checkpoint).parent / "best_threshold.json")
        save_threshold_curve(val_sweep.sweep(), out_dir / "val_threshold_curve.png",
                             threshold, "Validation set: threshold selection")
        print(f"  best validation F1 {best['f1']:.4f} at threshold {threshold:.2f}")
    else:
        threshold, source = resolve_threshold(args, checkpoint, args.checkpoint)

    print(f"Threshold  : {threshold:.2f}  ({source})")

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    dataset = create_test_dataset(use_canny) if args.split == "test" else create_validation_dataset(use_canny)
    print(f"\nEvaluating {args.split} set: {len(dataset)} full-resolution images")

    sweep, rows = run_split(model, dataset, device, threshold, args.tta,
                            args.num_vis, out_dir, args.split)

    micro = sweep.metrics_at(threshold)
    macro = {k: float(np.mean([r[k] for r in rows])) for k in ("precision", "recall", "f1", "iou")}
    oracle = sweep.best("f1")

    print("\n" + "=" * 60)
    print(f"FINAL {args.split.upper()} RESULTS (dataset-level, threshold {threshold:.2f})")
    print("=" * 60)
    for key in ("accuracy", "precision", "recall", "f1", "iou", "kappa"):
        print(f"{key.capitalize():<10}: {micro[key]:.4f}")
    print(f"\nTP: {micro['tp']:,} | TN: {micro['tn']:,} | FP: {micro['fp']:,} | FN: {micro['fn']:,}")
    print(f"\nPer-image mean F1 (macro, analysis only): {macro['f1']:.4f}")
    if args.split == "test":
        print(f"(For reference only - NOT a valid result: best possible test F1 "
              f"{oracle['f1']:.4f} at t={oracle['threshold']:.2f})")

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    tag = args.split
    save_confusion_matrix(micro["tp"], micro["tn"], micro["fp"], micro["fn"],
                          str(out_dir / "confusion_matrix.png"))
    save_metric_summary(micro, str(out_dir / "metrics_summary.png"))
    save_text_report(micro, str(out_dir / f"{tag}_metrics.txt"), test_samples=len(dataset))
    save_json_report(micro, str(out_dir / f"{tag}_metrics.json"), test_samples=len(dataset))
    save_threshold_curve(sweep.sweep(), out_dir / f"{tag}_threshold_curve.png", threshold,
                         f"{tag.capitalize()} set: metrics vs threshold")

    with open(out_dir / f"{tag}_per_image_metrics.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "input": "rgb+canny" if use_canny else "rgb",
        "tta": args.tta,
        "tile_overlap": args.overlap,
        "threshold": threshold,
        "threshold_source": source,
        "dataset_level": micro,
        "per_image_mean": macro,
        "num_images": len(dataset),
    }
    with open(out_dir / f"{tag}_summary.json", "w") as file:
        json.dump(summary, file, indent=4)

    print(f"\nResults saved to {out_dir}")
    print("Evaluation complete. ✅")


if __name__ == "__main__":
    main()
