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