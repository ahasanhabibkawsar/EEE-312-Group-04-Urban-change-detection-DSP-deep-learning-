"""
Evaluation metrics for binary change detection.

Two ways of averaging
---------------------

Dataset-level ("micro", the standard for LEVIR-CD papers)
    Sum TP / FP / FN / TN over ALL pixels of ALL images, then compute
    Precision, Recall, F1, IoU once. Use ConfusionAccumulator.
    With this averaging IoU = F1 / (2 - F1) holds exactly.

Per-image ("macro")
    Compute the metrics for each image and average them. Useful for
    error analysis, but it is NOT comparable with published numbers,
    and it is inflated by images without any change.

The empty-image rule ("Zero-F1 fix")
------------------------------------
For a single image with no changed pixels in the ground truth AND no
predicted change, TP = FP = FN = 0 and F1 = 0 / 0 is undefined. v1
defined it as 1.0 (a perfect "nothing changed" answer); otherwise every
correctly empty prediction would score 0 and drag the per-image average
down. This only affects PER-IMAGE metrics. It has no effect on
dataset-level metrics (the sums are never all zero) and none on
training, because metrics are not part of the loss / gradient.
"""

import numpy as np
import torch


def _default_threshold():
    from config import load_best_threshold
    return load_best_threshold()


# =========================================================
# Basic formulas
# =========================================================

def calculate_confusion_matrix(predictions, targets):
    predictions = predictions.bool()
    targets = targets > 0.5

    tp = (predictions & targets).sum().item()
    tn = (~predictions & ~targets).sum().item()
    fp = (predictions & ~targets).sum().item()
    fn = (~predictions & targets).sum().item()

    return int(tp), int(tn), int(fp), int(fn)


def calculate_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def calculate_recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def calculate_f1(precision, recall):
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def calculate_iou(tp, fp, fn):
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0


def calculate_kappa(tp, tn, fp, fn):
    """Cohen's kappa (agreement beyond chance)."""

    total = tp + tn + fp + fn
    if total == 0:
        return 0.0

    po = (tp + tn) / total
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def metrics_from_counts(tp, tn, fp, fn, empty_value=1.0):
    """
    All metrics from confusion-matrix counts.

    empty_value : value used for P / R / F1 / IoU when TP = FP = FN = 0
                  (an image with no change that was correctly predicted
                  as unchanged). See the module docstring.
    """

    tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)
    total = tp + tn + fp + fn

    if (tp + fp + fn) == 0:
        precision = recall = f1 = iou = float(empty_value)
    else:
        precision = calculate_precision(tp, fp)
        recall = calculate_recall(tp, fn)
        f1 = calculate_f1(precision, recall)
        iou = calculate_iou(tp, fp, fn)

    return {
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "kappa": calculate_kappa(tp, tn, fp, fn),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def calculate_metrics(logits, targets, threshold=None, from_probabilities=False):
    """
    Metrics for a tensor of logits (or probabilities) vs. binary targets.

    All pixels passed in are pooled, so passing the whole test set gives
    dataset-level metrics and passing one image gives per-image metrics.
    """

    if threshold is None:
        threshold = _default_threshold()

    probabilities = logits if from_probabilities else torch.sigmoid(logits)
    predictions = probabilities >= threshold

    return metrics_from_counts(*calculate_confusion_matrix(predictions, targets))


# =========================================================
# Accumulators
# =========================================================

class ConfusionAccumulator:
    """
    Dataset-level (micro) metrics accumulated image by image.
    """

    def __init__(self, threshold=None):
        self.threshold = _default_threshold() if threshold is None else threshold
        self.reset()

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0

    def update(self, probabilities, targets):
        tp, tn, fp, fn = calculate_confusion_matrix(probabilities >= self.threshold, targets)
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn
        return metrics_from_counts(tp, tn, fp, fn)

    def compute(self):
        return metrics_from_counts(self.tp, self.tn, self.fp, self.fn)


class ThresholdSweep:
    """
    Dataset-level metrics for MANY thresholds in one pass.

    Probabilities are binned into 1000 bins separately for changed and
    unchanged ground-truth pixels. For any threshold t = k / 1000:

        TP(t) = #changed pixels with p >= t
        FP(t) = #unchanged pixels with p >= t

    so the full precision-recall curve costs one histogram per image.
    """

    BINS = 1000

    def __init__(self):
        self.pos_hist = np.zeros(self.BINS, dtype=np.int64)
        self.neg_hist = np.zeros(self.BINS, dtype=np.int64)

    def update(self, probabilities, targets):
        p = probabilities.detach().cpu().double().reshape(-1)   # (MPS has no float64)
        t = (targets.detach().cpu().reshape(-1) > 0.5)

        idx = torch.clamp((p * self.BINS).long(), 0, self.BINS - 1)

        self.pos_hist += torch.bincount(idx[t], minlength=self.BINS).numpy()
        self.neg_hist += torch.bincount(idx[~t], minlength=self.BINS).numpy()

    def counts_at(self, threshold):
        k = int(round(threshold * self.BINS))
        k = min(max(k, 0), self.BINS)

        tp = int(self.pos_hist[k:].sum())
        fn = int(self.pos_hist[:k].sum())
        fp = int(self.neg_hist[k:].sum())
        tn = int(self.neg_hist[:k].sum())
        return tp, tn, fp, fn

    def metrics_at(self, threshold):
        return metrics_from_counts(*self.counts_at(threshold))

    def sweep(self, thresholds=None):
        if thresholds is None:
            thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)

        return [
            {"threshold": float(t), **self.metrics_at(float(t))}
            for t in thresholds
        ]

    def best(self, metric="f1", thresholds=None):
        results = self.sweep(thresholds)
        return max(results, key=lambda r: r[metric])
