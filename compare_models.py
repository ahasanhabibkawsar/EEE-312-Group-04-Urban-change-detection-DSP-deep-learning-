#!/usr/bin/env python3
"""
Compare the old (v1) and new (v2) models, the 3- vs 4-channel inputs and
two v1 + v2 ensembles on three datasets.

    python compare_models.py            # full run (about 30-40 min on an M-series Mac)
    python compare_models.py --quick    # every 4th image (about 10 min) for a first look

Methods
    v1       old prototype: raw RGB (0-255) + Canny, every image resized to 256 x 256
    v2       final model: standardised RGB + Canny (4 channels), native resolution
    v2rgb    final architecture, RGB only (3 channels)
    ens_avg  average of v1 and v2 probabilities
    ens_max  pixel-wise maximum (union) of v1 and v2

Datasets
    LEVIR-CD  test (128 x 1024 px, 0.5 m/px, labelled) - thresholds chosen on LEVIR-CD val
    WHU-CD    test (512 px tiles, 0.3 m/px, Christchurch NZ, labelled) - never used for training
    Purbachal (256 px tiles, Dhaka, NO labels) - predicted change area + figures

All thresholds are selected on LEVIR-CD validation only; WHU-CD and
Purbachal are pure out-of-domain tests.

Outputs (comparison_results/):
    results.json, results.md      metrics tables
    per_image_<dataset>.csv       per-image numbers
    fig_<dataset>_*.jpg           visual comparisons (red = predicted change, drawn on the BEFORE image)
    ../checkpoints/ensemble_thresholds.json  thresholds used by the GUI
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
from inference.multi_model import ModelZoo, METHODS, V2_RGB_CHECKPOINT, ENSEMBLE_FILE
from inference.predictor import get_device, load_image
from training.metrics import metrics_from_counts

ROOT = Path(config.PROJECT_ROOT)
LEVIR = ROOT / "data" / "LEVIR-CD"
WHU = ROOT / "data" / "WHU-CD" / "test"
PURBACHAL = ROOT / "data" / "LEVIR-CD_purbachal" / "test"

SHORT = {"v1": "v1 (old)", "v2": "v2 RGB+Canny", "v2rgb": "v2 RGB only",
         "ens_avg": "Ensemble avg", "ens_max": "Ensemble max"}


# ------------------------------------------------------------------ helpers

class Sweep:
    """Dataset-level metrics for every threshold (1000-bin histograms)."""

    BINS = 1000

    def __init__(self):
        self.pos = np.zeros(self.BINS, np.int64)
        self.neg = np.zeros(self.BINS, np.int64)

    def update(self, prob, gt):
        idx = np.clip((prob * self.BINS).astype(np.int64), 0, self.BINS - 1).ravel()
        g = gt.ravel()
        self.pos += np.bincount(idx[g], minlength=self.BINS)
        self.neg += np.bincount(idx[~g], minlength=self.BINS)

    def at(self, t):
        k = int(round(t * self.BINS))
        tp, fn = int(self.pos[k:].sum()), int(self.pos[:k].sum())
        fp, tn = int(self.neg[k:].sum()), int(self.neg[:k].sum())
        return metrics_from_counts(tp, tn, fp, fn)

    def best(self):
        ts = np.round(np.arange(0.05, 0.951, 0.01), 2)
        return max(({"threshold": float(t), **self.at(t)} for t in ts), key=lambda r: r["f1"])


def pairs(folder, ext, limit_step=1):
    a = sorted((folder / "A").glob(f"*{ext}"))
    out = []
    for p in a[::limit_step]:
        b = folder / "B" / p.name
        lab = folder / "label" / p.name
        if b.exists():
            out.append((p, b, lab if lab.exists() else None))
    return out


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return m > 127


def overlay(before, mask, color=(255, 0, 0)):
    """Predicted change on the BEFORE image: solid red fill + outline."""

    out = before.copy()
    m = mask.astype(bool)
    out[m] = (0.35 * out[m] + 0.65 * np.array(color)).astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 1)
    return out


def f1_of(pred, gt):
    tp = np.logical_and(pred, gt).sum(); fp = np.logical_and(pred, ~gt).sum(); fn = np.logical_and(~pred, gt).sum()
    return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def save_grid(rows, col_titles, path, cell=2.6):
    n_r, n_c = len(rows), len(col_titles)
    fig, axes = plt.subplots(n_r, n_c, figsize=(cell * n_c, cell * n_r + 0.4), squeeze=False)
    for r, row in enumerate(rows):
        for c, (img, sub) in enumerate(row):
            ax = axes[r][c]
            if img is None:
                ax.text(0.5, 0.5, "no ground truth", ha="center", va="center", fontsize=9, color="#555")
                ax.set_facecolor("#eeeeee")
            else:
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1 if img.ndim == 2 else None)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10, fontweight="bold")
            if sub:
                ax.set_xlabel(sub, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ main pass

def run_dataset(zoo, name, items, entries, tta, has_gt, thresholds, out_dir, log_every=10):
    """
    entries: list of (key, method, v2_scale). Returns metrics dict and per-image rows.
    """

    sweeps = {k: Sweep() for k, _, _ in entries}
    rows = []
    t0 = time.time()

    for i, (pa, pb, pl) in enumerate(items, 1):
        before, after = load_image(pa), load_image(pb)
        if after.shape != before.shape:
            after = cv2.resize(after, (before.shape[1], before.shape[0]))
        gt = read_mask(pl) if (has_gt and pl is not None) else None
        cache = {}
        row = {"image": pa.name}
        if gt is not None:
            row["gt_change_%"] = round(100 * gt.mean(), 3)
        masks = {}
        for key, method, scale in entries:
            prob = zoo.predict(method, before, after, tta=tta, scale=scale, cache=cache)
            pred = prob >= thresholds.get(key, thresholds[method])
            masks[key] = pred
            row[f"{key}_change_%"] = round(100 * pred.mean(), 3)
            if gt is not None:
                sweeps[key].update(prob, gt)
                row[f"{key}_f1"] = round(f1_of(pred, gt), 4)
        if "v1" in masks and "v2" in masks:
            inter = np.logical_and(masks["v1"], masks["v2"]).sum(); uni = np.logical_or(masks["v1"], masks["v2"]).sum()
            row["v1_v2_agreement_iou"] = round(inter / uni, 4) if uni else 1.0
        rows.append(row)

        if i % log_every == 0 or i == len(items):
            el = time.time() - t0
            print(f"  [{name}] {i}/{len(items)}  {el / 60:.1f} min elapsed, ~{el / i * (len(items) - i) / 60:.1f} min left", flush=True)

    with open(out_dir / f"per_image_{name}.csv", "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    summary = {}
    for key, method, scale in entries:
        s = {"method": method, "v2_scale": scale, "threshold": thresholds.get(key, thresholds[method])}
        s["mean_pred_change_%"] = float(np.mean([r[f"{key}_change_%"] for r in rows]))
        s["images_with_detection_%"] = float(np.mean([r[f"{key}_change_%"] >= 0.5 for r in rows]) * 100)
        if has_gt:
            m = sweeps[key].at(s["threshold"])
            b = sweeps[key].best()
            s.update({k: m[k] for k in ("precision", "recall", "f1", "iou", "accuracy", "kappa", "tp", "fp", "fn", "tn")})
            s["oracle_threshold"], s["oracle_f1"] = b["threshold"], b["f1"]
            s["per_image_mean_f1"] = float(np.mean([r[f"{key}_f1"] for r in rows]))
        summary[key] = s
    if rows and "v1_v2_agreement_iou" in rows[0]:
        summary["_v1_v2_mean_agreement_iou"] = float(np.mean([r["v1_v2_agreement_iou"] for r in rows]))
    return summary, rows, sweeps


def figure_for(zoo, name, items, entries, tta, thresholds, out_dir, fname, has_gt, extra_note=""):
    col_titles = ["Before (T1)", "After (T2)", "Ground truth"] + [lbl for _, _, _, lbl in entries]
    rows = []
    for pa, pb, pl in items:
        before, after = load_image(pa), load_image(pb)
        if after.shape != before.shape:
            after = cv2.resize(after, (before.shape[1], before.shape[0]))
        gt = read_mask(pl) if (has_gt and pl is not None) else None
        cache = {}
        r = [(before, pa.stem), (after, extra_note), (gt.astype(np.float32) if gt is not None else None,
                                                      f"{100 * gt.mean():.1f} % changed" if gt is not None else "")]
        for key, method, scale, _ in entries:
            prob = zoo.predict(method, before, after, tta=tta, scale=scale, cache=cache)
            pred = prob >= thresholds.get(key, thresholds[method])
            sub = f"F1 {f1_of(pred, gt):.2f}" if gt is not None else f"{100 * pred.mean():.1f} % changed"
            r.append((overlay(before, pred), sub))
        rows.append(r)
    save_grid(rows, col_titles, out_dir / fname)
    print(f"  saved {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="levir,whu,purbachal")
    ap.add_argument("--quick", action="store_true", help="every 4th image")
    ap.add_argument("--out-dir", default="comparison_results")
    ap.add_argument("--no-levir-tta", action="store_true")
    args = ap.parse_args()

    step = 4 if args.quick else 1
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)
    device = get_device()
    zoo = ZooWithLog(device)
    print(f"Device: {device}. Available methods: {zoo.available()}")
    has_rgb = V2_RGB_CHECKPOINT.exists()

    from inference.multi_model import default_thresholds
    thresholds = default_thresholds()
    results = {"quick": args.quick, "thresholds": {}, "datasets": {}}
    res_path = out_dir / "results.json"
    datasets = args.datasets.split(",")
    levir_tta = not args.no_levir_tta

    # ---------------- 1. thresholds on LEVIR-CD validation (v1, v2, ensembles)
    if "levir" in datasets:
        print("\n== LEVIR-CD validation: choosing thresholds")
        val = pairs(LEVIR / "val", ".png", step)
        ent = [("v1", "v1", 1.0), ("v2", "v2", 1.0), ("ens_avg", "ens_avg", 1.0), ("ens_max", "ens_max", 1.0)]
        _, _, sweeps = run_dataset(zoo, "levir_val", val, ent, levir_tta, True, thresholds, out_dir)
        for k in ("v1", "ens_avg", "ens_max"):
            thresholds[k] = sweeps[k].best()["threshold"]
        results["thresholds"]["val_best_f1"] = {k: sweeps[k].best()["f1"] for k in sweeps}
        ENSEMBLE_FILE.write_text(json.dumps({k: thresholds[k] for k in ("v1", "ens_avg", "ens_max")}, indent=2))
        print("  thresholds:", {k: thresholds[k] for k in ("v1", "v2", "v2rgb", "ens_avg", "ens_max")})
    results["thresholds"]["used"] = thresholds

    main_entries = [("v1", "v1", 1.0), ("v2", "v2", 1.0)] + ([("v2rgb", "v2rgb", 1.0)] if has_rgb else []) + \
                   [("ens_avg", "ens_avg", 1.0), ("ens_max", "ens_max", 1.0)]

    # ---------------- 2. LEVIR-CD test
    if "levir" in datasets:
        print("\n== LEVIR-CD test (in-domain)")
        items = pairs(LEVIR / "test", ".png", step)
        summ, _, _ = run_dataset(zoo, "levir_test", items, main_entries, levir_tta, True, thresholds, out_dir)
        results["datasets"]["levir_test"] = {"n": len(items), "tta": levir_tta, "methods": summ}
        res_path.write_text(json.dumps(results, indent=2))
        pick = [p for p in pairs(LEVIR / "test", ".png") if p[0].stem in ("test_102", "test_100", "test_103")]
        figure_for(zoo, "levir", pick, [(k, m, s, SHORT[k]) for k, m, s in main_entries], levir_tta,
                   thresholds, out_dir, "fig_levir.jpg", True)

    # ---------------- 3. WHU-CD test (out-of-domain, labelled, 0.3 m/px)
    if "whu" in datasets and (WHU / "A").exists():
        print("\n== WHU-CD test (out-of-domain, 0.3 m/px -> v2 resampled x0.6)")
        items = pairs(WHU, ".tif", step)
        ent = [(k, m, 0.6) for k, m, _ in main_entries] + [("v2_native", "v2", 1.0)]
        th = dict(thresholds, v2_native=thresholds["v2"])
        summ, rows, _ = run_dataset(zoo, "whu_test", items, ent, False, True, th, out_dir, log_every=50)
        results["datasets"]["whu_test"] = {"n": len(items), "tta": False, "methods": summ}
        res_path.write_text(json.dumps(results, indent=2))
        rows = sorted(rows, key=lambda r: -r.get("gt_change_%", 0))[:4]
        pick = [p for p in items if p[0].name in {r["image"] for r in rows}]
        figure_for(zoo, "whu", pick, [(k, m, 0.6, SHORT[k]) for k, m, _ in main_entries], False,
                   thresholds, out_dir, "fig_whu.jpg", True)

    # ---------------- 4. Purbachal, Dhaka (no labels)
    if "purbachal" in datasets and (PURBACHAL / "A").exists():
        print("\n== Purbachal, Dhaka (no ground truth)")
        items = pairs(PURBACHAL, ".png", step)
        ent = main_entries + [("v2_x2", "v2", 2.0), ("v2_x4", "v2", 4.0)]
        th = dict(thresholds, v2_x2=thresholds["v2"], v2_x4=thresholds["v2"])
        summ, rows, _ = run_dataset(zoo, "purbachal", items, ent, False, False, th, out_dir, log_every=25)
        results["datasets"]["purbachal"] = {"n": len(items), "tta": False, "methods": summ}
        res_path.write_text(json.dumps(results, indent=2))
        top = sorted(rows, key=lambda r: -r["v1_change_%"])[:5]
        pick = [p for p in items if p[0].name in {r["image"] for r in top}]
        figure_for(zoo, "purbachal", pick, [(k, m, s, SHORT[k]) for k, m, s in main_entries], False,
                   thresholds, out_dir, "fig_purbachal.jpg", False)
        figure_for(zoo, "purbachal", pick, [("v1", "v1", 1.0, "v1 (old)"), ("v2", "v2", 1.0, "v2 ×1 (native)"),
                                            ("v2_x2", "v2", 2.0, "v2 ×2 upsampled"), ("v2_x4", "v2", 4.0, "v2 ×4 upsampled")],
                   False, th, out_dir, "fig_purbachal_scale.jpg", False)

    # ---------------- report
    lines = ["# Model comparison", "", f"Thresholds (LEVIR-CD validation): " +
             ", ".join(f"{k} {v:.2f}" for k, v in thresholds.items() if k in METHODS), ""]
    for ds, d in results["datasets"].items():
        lines += [f"## {ds} (n = {d['n']}, TTA {'on' if d['tta'] else 'off'})", ""]
        has = "f1" in next(iter(v for k, v in d["methods"].items() if not k.startswith("_")))
        if has:
            lines += ["| Method | t | Precision | Recall | F1 | IoU | best-t F1 | per-image F1 |", "|---|---|---|---|---|---|---|---|"]
            for k, s in d["methods"].items():
                if k.startswith("_"): continue
                lines.append(f"| {k} | {s['threshold']:.2f} | {100*s['precision']:.2f} | {100*s['recall']:.2f} | "
                             f"**{100*s['f1']:.2f}** | {100*s['iou']:.2f} | {100*s['oracle_f1']:.2f} @ {s['oracle_threshold']:.2f} | {100*s['per_image_mean_f1']:.2f} |")
        else:
            lines += ["| Method | t | mean predicted change | images with ≥ 0.5 % change |", "|---|---|---|---|"]
            for k, s in d["methods"].items():
                if k.startswith("_"): continue
                lines.append(f"| {k} | {s['threshold']:.2f} | {s['mean_pred_change_%']:.2f} % | {s['images_with_detection_%']:.1f} % |")
        if "_v1_v2_mean_agreement_iou" in d["methods"]:
            lines.append(f"\nv1–v2 mask agreement (mean IoU): {d['methods']['_v1_v2_mean_agreement_iou']:.3f}")
        lines.append("")
    (out_dir / "results.md").write_text("\n".join(lines))
    res_path.write_text(json.dumps(results, indent=2))
    print("\n" + "\n".join(lines))
    print(f"\nSaved to {out_dir}/")


class ZooWithLog(ModelZoo):
    def get(self, name):
        if name not in self.models:
            print(f"  loading {name} ...", flush=True)
        return super().get(name)


if __name__ == "__main__":
    main()
