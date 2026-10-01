#!/usr/bin/env python3
"""
Save one comparison figure for EVERY image pair in a folder.

Each figure:  row 1  Before | After | Ground truth | v1 | v2 4-ch | v2 3-ch | Ens. avg | Ens. max
                     (red = detected change, drawn on the Before image by default)
              row 2  change-probability map of each model

    python save_all_samples.py                                   # Purbachal (data/LEVIR-CD_purbachal/test)
    python save_all_samples.py --data-dir data/LEVIR-CD/test     # LEVIR-CD test (128 figures)
    python save_all_samples.py --data-dir data/WHU-CD/test --limit 50
    python save_all_samples.py --overlay after                   # draw on the After image instead
    python save_all_samples.py --tta                             # slower, slightly more accurate

Folder layout: <data-dir>/A (before), <data-dir>/B (after), optional <data-dir>/label.
Figures go to comparison_results/samples_<name>/ ; a summary CSV is written next to them.
Thresholds are the LEVIR-CD-validation thresholds used everywhere else.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from compare_models import ZooWithLog, overlay, read_mask, f1_of, SHORT
from inference.multi_model import default_thresholds
from inference.predictor import get_device, load_image

ROOT = Path(config.PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/LEVIR-CD_purbachal/test")
    ap.add_argument("--name", default=None, help="output folder suffix (default: from data-dir)")
    ap.add_argument("--overlay", choices=["before", "after"], default="before")
    ap.add_argument("--scale", type=float, default=1.0, help="v2 resampling (WHU-CD 0.3 m/px: use 0.6)")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = (ROOT / args.data_dir) if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    name = args.name or "_".join(data.parts[-2:]).replace("LEVIR-CD_", "").replace("-", "").lower()
    out = ROOT / "comparison_results" / f"samples_{name}"
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in (data / "A").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    files = [p for p in files if (data / "B" / p.name).exists()]
    if args.limit:
        files = files[:args.limit]

    zoo = ZooWithLog(get_device())
    methods = [m for m in ["v1", "v2", "v2rgb", "ens_avg", "ens_max"] if m in zoo.available()]
    th = default_thresholds()
    print(f"{len(files)} pairs -> {out}")

    rows = []
    for i, pa in enumerate(files, 1):
        before = load_image(pa)
        after = load_image(data / "B" / pa.name)
        if after.shape != before.shape:
            after = cv2.resize(after, (before.shape[1], before.shape[0]))
        lab = data / "label" / pa.name
        gt = read_mask(lab) if lab.exists() else None
        base = before if args.overlay == "before" else after

        cache, probs = {}, {}
        for m in methods:
            probs[m] = zoo.predict(m, before, after, tta=args.tta, scale=args.scale, cache=cache)

        n = 3 + len(methods)
        fig, axes = plt.subplots(2, n, figsize=(2.7 * n, 5.9))
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, 0].imshow(before); axes[0, 0].set_title("Before (T1)", fontsize=10, fontweight="bold")
        axes[0, 1].imshow(after); axes[0, 1].set_title("After (T2)", fontsize=10, fontweight="bold")
        axes[0, 2].set_title("Ground truth", fontsize=10, fontweight="bold")
        if gt is not None:
            axes[0, 2].imshow(gt, cmap="gray"); axes[0, 2].set_xlabel(f"{100 * gt.mean():.1f} % changed", fontsize=9)
        else:
            axes[0, 2].text(0.5, 0.5, "no ground truth", ha="center", va="center", color="#666")
        for ax in axes[1, :3]:
            ax.axis("off")
        axes[1, 0].text(0.0, 0.55, f"{pa.stem}\nred = detected change\n(on {args.overlay} image)\n"
                        f"bottom: probability", fontsize=9, va="center")

        row = {"image": pa.name}
        if gt is not None:
            row["gt_change_%"] = round(100 * gt.mean(), 2)
        for j, m in enumerate(methods):
            pred = probs[m] >= th[m]
            sub = f"{100 * pred.mean():.1f} % changed"
            row[f"{m}_change_%"] = round(100 * pred.mean(), 2)
            if gt is not None:
                f1 = f1_of(pred, gt); row[f"{m}_f1"] = round(f1, 4)
                sub = f"F1 {f1:.2f} · " + sub
            axes[0, 3 + j].imshow(overlay(base, pred))
            axes[0, 3 + j].set_title(SHORT[m], fontsize=10, fontweight="bold")
            axes[0, 3 + j].set_xlabel(sub, fontsize=9)
            axes[1, 3 + j].imshow(probs[m], cmap="viridis", vmin=0, vmax=1)
            axes[1, 3 + j].set_xlabel(f"t = {th[m]:.2f}", fontsize=9)
        plt.tight_layout()
        fig.savefig(out / f"{pa.stem}.jpg", dpi=100)
        plt.close(fig)
        rows.append(row)
        print(f"  {i}/{len(files)}  {pa.name}", flush=True)

    with open(out / "summary.csv", "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"Saved {len(rows)} figures + summary.csv to {out}")


if __name__ == "__main__":
    main()
