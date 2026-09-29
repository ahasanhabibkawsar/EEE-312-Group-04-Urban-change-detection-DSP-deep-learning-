"""
============================================================
URBAN CHANGE DETECTION - FULL TEST-SET INFERENCE (v2)
============================================================

Runs the model on every image pair of a split, at native resolution,
and reports metrics for BOTH the raw network output and the
post-processed mask (opening + small-region removal + polygons), so
the effect of post-processing is measured instead of assumed.

Works on any folder with A/ and B/ sub-folders; label/ is optional
(e.g. new imagery such as the Purbachal pairs).

    python test_all_inference.py
    python test_all_inference.py --tta
    python test_all_inference.py --data-dir data/LEVIR-CD_purbachal/test --out-dir purbachal_results
============================================================
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from inference.predictor import get_device, load_model, run_inference, postprocess_mask
from training.metrics import metrics_from_counts
from evaluation.visualization import error_map

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(config.TEST_DIR))
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_PATH))
    parser.add_argument("--out-dir", type=str, default=str(config.PROJECT_ROOT / "full_inference_results"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Resampling factor = source GSD / 0.5 m")
    parser.add_argument("--num-vis", type=int, default=10)
    return parser.parse_args()


def counts(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5
    return (np.logical_and(pred, gt).sum(), np.logical_and(~pred, ~gt).sum(),
            np.logical_and(pred, ~gt).sum(), np.logical_and(~pred, gt).sum())


def save_visualization(result, gt, clean, path, title):
    panels = [(result["before"], "Before"), (result["after"], "After")]
    if gt is not None:
        panels.append((gt, "Ground truth"))
    panels += [(result["probability"], "Probability"), (clean, "Post-processed mask")]
    if gt is not None:
        panels.append((error_map(clean, gt), "Errors: white TP, red FP, cyan FN"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.4))
    for axis, (image, name) in zip(axes, panels):
        if image.ndim == 2:
            axis.imshow(image, cmap="viridis" if name == "Probability" else "gray", vmin=0, vmax=1)
        else:
            axis.imshow(image)
        axis.set_title(name)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model = load_model(device, args.checkpoint)
    threshold = args.threshold if args.threshold is not None else config.load_best_threshold(
        Path(args.checkpoint).parent / "best_threshold.json")

    files = sorted(p.name for p in (data_dir / "A").iterdir()
                   if p.suffix.lower() in IMAGE_EXTENSIONS and (data_dir / "B" / p.name).exists())
    label_dir = data_dir / "label"

    print("=" * 60)
    print(f"FULL INFERENCE: {len(files)} pairs from {data_dir}")
    print(f"threshold {threshold:.2f} | TTA {args.tta} | scale {args.scale} | device {device}")
    print("=" * 60)

    rows = []
    totals = {"raw": np.zeros(4, dtype=np.int64), "post": np.zeros(4, dtype=np.int64)}
    start = time.time()

    for index, name in enumerate(files):
        result = run_inference(model, data_dir / "A" / name, data_dir / "B" / name, device,
                               threshold=threshold, tta=args.tta, scale=args.scale)
        result["before"] = result["before"].astype(np.float32) / 255.0
        result["after"] = result["after"].astype(np.float32) / 255.0

        raw = result["mask"]
        clean = postprocess_mask(raw)

        row = {"filename": name,
               "pred_change_percent": 100.0 * float(raw.mean()),
               "post_change_percent": 100.0 * float(clean.mean())}

        gt = None
        if (label_dir / name).exists():
            gt = cv2.imread(str(label_dir / name), cv2.IMREAD_GRAYSCALE)
            if gt.shape != raw.shape:
                gt = cv2.resize(gt, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt = (gt > 127).astype(np.float32)

            for key, mask in (("raw", raw), ("post", clean)):
                c = counts(mask, gt)
                totals[key] += np.array(c, dtype=np.int64)
                m = metrics_from_counts(*c)
                for metric in ("f1", "iou", "precision", "recall"):
                    row[f"{key}_{metric}"] = m[metric]

            # Columns read by generate_report.py (post-processed mask, as in v1)
            post = metrics_from_counts(*counts(clean, gt))
            row.update({k: post[k] for k in ("accuracy", "precision", "recall", "f1", "iou",
                                             "tp", "tn", "fp", "fn")})
            row["predicted_pixels"] = int(clean.sum())
            row["actual_pixels"] = int(gt.sum())
            row["predicted_change_percent"] = 100.0 * float(clean.mean())
            row["actual_change_percent"] = 100.0 * float(gt.mean())

        rows.append(row)

        if index < args.num_vis:
            save_visualization(result, gt, clean, vis_dir / f"{Path(name).stem}_inference.png", name)

        f1_text = f"F1 raw {row['raw_f1']:.3f} / post {row['post_f1']:.3f}" if gt is not None else ""
        print(f"{index + 1:3d}/{len(files)} | {name:<20} | {f1_text}", flush=True)

    print(f"\nTotal time: {time.time() - start:.1f}s")

    preferred = ["filename", "accuracy", "precision", "recall", "f1", "iou", "tp", "tn", "fp", "fn",
                 "predicted_pixels", "actual_pixels", "predicted_change_percent", "actual_change_percent"]
    extra = sorted({key for row in rows for key in row} - set(preferred))
    fieldnames = [k for k in preferred if any(k in row for row in rows)] + extra
    with open(out_dir / "per_image_metrics.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    overall = {"threshold": threshold, "tta": args.tta, "num_images": len(files)}
    if any("raw_f1" in row for row in rows):
        for key in ("raw", "post"):
            overall[f"{key}_dataset_level"] = metrics_from_counts(*totals[key])
            overall[f"{key}_per_image_mean_f1"] = float(np.mean([r[f"{key}_f1"] for r in rows if f"{key}_f1" in r]))

        # Keys used by generate_report.py (post-processed, per-image means + dataset-level)
        labelled = [r for r in rows if "f1" in r]
        for metric in ("accuracy", "precision", "recall", "f1", "iou"):
            overall[metric] = float(np.mean([r[metric] for r in labelled]))
            overall[f"micro_{metric}"] = overall["post_dataset_level"][metric]
        overall.update({k: overall["post_dataset_level"][k] for k in ("tp", "tn", "fp", "fn")})

        print("\n" + "=" * 60)
        for key, name in (("raw", "Raw network output"), ("post", "Post-processed")):
            m = overall[f"{key}_dataset_level"]
            print(f"{name:<20}: F1 {m['f1']:.4f} | IoU {m['iou']:.4f} | P {m['precision']:.4f} | R {m['recall']:.4f}")
        print("=" * 60)

    with open(out_dir / "overall_metrics.json", "w") as file:
        json.dump(overall, file, indent=4)

    print(f"Results saved to {out_dir}")
    print("FULL INFERENCE COMPLETE ✅")


if __name__ == "__main__":
    main()
