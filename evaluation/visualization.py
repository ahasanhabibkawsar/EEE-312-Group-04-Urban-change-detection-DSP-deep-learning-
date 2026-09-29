"""
============================================================
EVALUATION VISUALIZATION
============================================================

Functions for creating:
    - Confusion matrix
    - Metric summary plot
    - Prediction comparison figures

============================================================
"""

import os

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    tp,
    tn,
    fp,
    fn,
    output_path
):
    """
    Create and save a pixel-level confusion matrix.

    Matrix layout:

                  Ground Truth
                No Change  Change

    Prediction
    No Change      TN        FN
    Change         FP        TP
    """

    matrix = np.array(
        [
            [tn, fn],
            [fp, tp]
        ],
        dtype=np.int64
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(
        matrix
    )

    axis.set_title(
        "Pixel-Level Confusion Matrix",
        fontsize=14
    )

    axis.set_xlabel(
        "Ground Truth"
    )

    axis.set_ylabel(
        "Prediction"
    )

    axis.set_xticks(
        [0, 1]
    )

    axis.set_yticks(
        [0, 1]
    )

    axis.set_xticklabels(
        [
            "No Change",
            "Change"
        ]
    )

    axis.set_yticklabels(
        [
            "No Change",
            "Change"
        ]
    )

    # --------------------------------------------------------
    # Add values
    # --------------------------------------------------------

    labels = [
        ["TN", "FN"],
        ["FP", "TP"]
    ]

    for row in range(2):

        for column in range(2):

            axis.text(
                column,
                row,
                f"{labels[row][column]}\n"
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                fontsize=12
            )

    figure.colorbar(
        image,
        ax=axis
    )

    figure.tight_layout()

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


# ============================================================
# METRIC SUMMARY
# ============================================================

def save_metric_summary(
    metrics,
    output_path
):
    """
    Create a bar chart containing the main
    test-set performance metrics.
    """

    names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "IoU"
    ]

    values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["iou"]
    ]

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    bars = axis.bar(
        names,
        values
    )

    axis.set_ylim(
        0,
        1.0
    )

    axis.set_ylabel(
        "Score"
    )

    axis.set_title(
        "Urban Change Detection - Test Performance"
    )

    # --------------------------------------------------------
    # Add values above bars
    # --------------------------------------------------------

    for bar, value in zip(
        bars,
        values
    ):

        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

    figure.tight_layout()

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


# ============================================================
# SAVE TRAINING VS TEST COMPARISON
# ============================================================

def save_validation_test_comparison(
    validation_metrics,
    test_metrics,
    output_path
):
    """
    Compare validation and test performance.
    """

    names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "IoU"
    ]

    validation_values = [
        validation_metrics["accuracy"],
        validation_metrics["precision"],
        validation_metrics["recall"],
        validation_metrics["f1"],
        validation_metrics["iou"]
    ]

    test_values = [
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
        test_metrics["iou"]
    ]

    x = np.arange(
        len(names)
    )

    width = 0.35

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.bar(
        x - width / 2,
        validation_values,
        width,
        label="Validation"
    )

    axis.bar(
        x + width / 2,
        test_values,
        width,
        label="Test"
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        names
    )

    axis.set_ylim(
        0,
        1.0
    )

    axis.set_ylabel(
        "Score"
    )

    axis.set_title(
        "Validation vs Test Performance"
    )

    axis.legend()

    figure.tight_layout()

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

# ============================================================
# v2 additions
# ============================================================

def error_map(prediction, ground_truth):
    """
    Colour-coded error map (H x W x 3, float in [0, 1]):

        TP -> white     FP -> red     FN -> cyan     TN -> black
    """

    prediction = np.asarray(prediction) > 0.5
    ground_truth = np.asarray(ground_truth) > 0.5

    colours = np.zeros(prediction.shape + (3,), dtype=np.float32)
    colours[prediction & ground_truth] = (1.0, 1.0, 1.0)
    colours[prediction & ~ground_truth] = (0.90, 0.15, 0.15)
    colours[~prediction & ground_truth] = (0.10, 0.80, 0.90)
    return colours


def save_prediction_panel(before_rgb, after_rgb, ground_truth, probability,
                          threshold, output_path, title="", metrics=None):
    """
    Six-panel figure: Before | After | Ground truth | Probability |
    Prediction | Error map (TP / FP / FN).
    """

    prediction = probability >= threshold

    figure, axes = plt.subplots(1, 6, figsize=(24, 4.4))

    panels = [
        (before_rgb, "Before (T1)", None),
        (after_rgb, "After (T2)", None),
        (ground_truth, "Ground truth", "gray"),
        (probability, "Change probability", "viridis"),
        (prediction, f"Prediction (t = {threshold:.2f})", "gray"),
        (error_map(prediction, ground_truth), "Errors: white TP, red FP, cyan FN", None),
    ]

    for axis, (image, name, cmap) in zip(axes, panels):
        if cmap is None:
            axis.imshow(image)
        else:
            axis.imshow(image, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(name, fontsize=11)
        axis.axis("off")

    if metrics is not None:
        title = (f"{title}   |   F1 {metrics['f1']:.3f}   IoU {metrics['iou']:.3f}   "
                 f"P {metrics['precision']:.3f}   R {metrics['recall']:.3f}")

    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    figure.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def save_threshold_curve(sweep_results, output_path, chosen_threshold=None, title=""):
    """
    Precision, Recall, F1 and IoU as a function of the decision threshold.
    """

    thresholds = [r["threshold"] for r in sweep_results]

    figure, axis = plt.subplots(figsize=(8, 5))

    for key, label in (("precision", "Precision"), ("recall", "Recall"),
                       ("f1", "F1"), ("iou", "IoU")):
        axis.plot(thresholds, [r[key] for r in sweep_results], label=label, linewidth=2)

    if chosen_threshold is not None:
        axis.axvline(chosen_threshold, color="gray", linestyle="--",
                     label=f"chosen t = {chosen_threshold:.2f}")

    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.3)
    axis.legend()
    axis.set_title(title or "Threshold sensitivity (dataset-level)")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
