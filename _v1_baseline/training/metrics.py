"""
Evaluation metrics for binary change detection.
"""

import torch

def calculate_confusion_matrix(predictions, targets):
    predictions = predictions.float()
    targets = targets.float()

    tp = ((predictions == 1) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()

    return tp, tn, fp, fn

def calculate_precision(tp, fp):
    denominator = tp + fp
    if denominator == 0: return 0.0
    return tp / denominator

def calculate_recall(tp, fn):
    denominator = tp + fn
    if denominator == 0: return 0.0
    return tp / denominator

def calculate_f1(precision, recall):
    denominator = precision + recall
    if denominator == 0: return 0.0
    return 2.0 * precision * recall / denominator

def calculate_iou(tp, fp, fn):
    denominator = tp + fp + fn
    if denominator == 0: return 0.0
    return tp / denominator

def calculate_metrics(logits, targets, threshold=0.45):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    tp, tn, fp, fn = calculate_confusion_matrix(predictions, targets)

    # --- THE ZERO-F1 MATH FIX ---
    if (tp + fp + fn) == 0:
        return {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "iou": 1.0,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }
    # ----------------------------

    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1 = calculate_f1(precision, recall)
    iou = calculate_iou(tp, fp, fn)

    total = tp + tn + fp + fn
    accuracy = 0.0 if total == 0 else (tp + tn) / total

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }