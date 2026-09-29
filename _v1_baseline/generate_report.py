"""
============================================================
URBAN CHANGE DETECTION
AUTOMATED EVALUATION REPORT
============================================================

Reads:
    full_inference_results/per_image_metrics.csv

Generates:
    evaluation_report/
        summary.txt
        metrics_summary.json
        plots/
            f1_distribution.png
            iou_distribution.png
            precision_recall.png
            metrics_comparison.png
        samples/
            best_01.txt
            ...
            worst_01.txt
            ...

This script does NOT retrain the model.
It only analyzes the results produced by
test_all_inference.py.
============================================================
"""

from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_ROOT / "full_inference_results"

CSV_FILE = INPUT_DIR / "per_image_metrics.csv"
JSON_FILE = INPUT_DIR / "overall_metrics.json"

REPORT_DIR = PROJECT_ROOT / "evaluation_report"
PLOTS_DIR = REPORT_DIR / "plots"
SAMPLES_DIR = REPORT_DIR / "samples"

SUMMARY_FILE = REPORT_DIR / "summary.txt"
SUMMARY_JSON = REPORT_DIR / "metrics_summary.json"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():
    """Create output directories."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    SAMPLES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CHECK INPUT FILES
# ============================================================

def check_input_files():
    """Check whether inference results exist."""

    print("=" * 60)
    print("CHECKING INFERENCE RESULTS")
    print("=" * 60)

    print(f"Input directory : {INPUT_DIR}")
    print(f"CSV file        : {CSV_FILE}")
    print(f"JSON file       : {JSON_FILE}")
    print()

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found:\n{INPUT_DIR}\n\n"
            "Run test_all_inference.py first."
        )

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{CSV_FILE}\n\n"
            "Run test_all_inference.py first."
        )

    print("Inference results found. ✅")
    print()


# ============================================================
# LOAD CSV
# ============================================================

def load_results():
    """
    Load per-image results from CSV.
    """

    print("=" * 60)
    print("LOADING PER-IMAGE RESULTS")
    print("=" * 60)

    results = []

    with open(
        CSV_FILE,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            # --------------------------------------------
            # Convert numerical fields
            # --------------------------------------------

            numeric_fields = [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "iou",
                "tp",
                "tn",
                "fp",
                "fn",
                "predicted_pixels",
                "actual_pixels",
                "predicted_change_percent",
                "actual_change_percent",
            ]

            for field in numeric_fields:

                if field in row:

                    if field in [
                        "tp",
                        "tn",
                        "fp",
                        "fn",
                        "predicted_pixels",
                        "actual_pixels",
                    ]:

                        row[field] = int(
                            float(row[field])
                        )

                    else:

                        row[field] = float(
                            row[field]
                        )

            results.append(row)

    print(
        f"Images loaded: {len(results)}"
    )

    print()

    if len(results) == 0:
        raise RuntimeError(
            "No image results found in CSV."
        )

    return results


# ============================================================
# BASIC STATISTICS
# ============================================================

def calculate_statistics(results):
    """
    Calculate mean, minimum, maximum and standard deviation
    for important metrics.
    """

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "iou",
    ]

    statistics = {}

    for metric in metric_names:

        values = np.array(
            [
                row[metric]
                for row in results
            ],
            dtype=np.float64
        )

        statistics[metric] = {
            "mean": float(
                np.mean(values)
            ),

            "std": float(
                np.std(values)
            ),

            "minimum": float(
                np.min(values)
            ),

            "maximum": float(
                np.max(values)
            ),

            "median": float(
                np.median(values)
            ),
        }

    return statistics


# ============================================================
# AGGREGATE CONFUSION MATRIX
# ============================================================

def calculate_confusion_matrix(results):
    """
    Aggregate TP/TN/FP/FN over all test images.
    """

    tp = sum(
        row["tp"]
        for row in results
    )

    tn = sum(
        row["tn"]
        for row in results
    )

    fp = sum(
        row["fp"]
        for row in results
    )

    fn = sum(
        row["fn"]
        for row in results
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


# ============================================================
# PIXEL METRICS
# ============================================================

def calculate_pixel_metrics(confusion):
    """
    Calculate metrics from aggregated pixels.
    """

    tp = confusion["TP"]
    tn = confusion["TN"]
    fp = confusion["FP"]
    fn = confusion["FN"]

    total = (
        tp
        + tn
        + fp
        + fn
    )

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


# ============================================================
# BEST / WORST SAMPLES
# ============================================================

def find_best_worst(results):
    """
    Find best and worst images based on F1-score.
    """

    sorted_results = sorted(
        results,
        key=lambda row: row["f1"]
    )

    best = list(
        reversed(
            sorted_results[-5:]
        )
    )

    worst = sorted_results[:5]

    return best, worst


# ============================================================
# F1 DISTRIBUTION
# ============================================================

def plot_f1_distribution(results):
    """
    Plot distribution of F1 scores.
    """

    values = [
        row["f1"]
        for row in results
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        values,
        bins=20,
        edgecolor="black"
    )

    plt.xlabel(
        "F1-score"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "F1-score Distribution on LEVIR-CD Test Set"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        PLOTS_DIR
        / "f1_distribution.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# IOU DISTRIBUTION
# ============================================================

def plot_iou_distribution(results):
    """
    Plot distribution of IoU scores.
    """

    values = [
        row["iou"]
        for row in results
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        values,
        bins=20,
        edgecolor="black"
    )

    plt.xlabel(
        "IoU"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "IoU Distribution on LEVIR-CD Test Set"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        PLOTS_DIR
        / "iou_distribution.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# PRECISION / RECALL
# ============================================================

def plot_precision_recall(results):
    """
    Plot precision vs recall for all images.
    """

    precision = [
        row["precision"]
        for row in results
    ]

    recall = [
        row["recall"]
        for row in results
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        recall,
        precision,
        alpha=0.7
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision vs Recall"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        PLOTS_DIR
        / "precision_recall.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# METRIC COMPARISON
# ============================================================

def plot_metric_comparison(statistics):
    """
    Plot average performance of the model.
    """

    names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "IoU",
    ]

    values = [
        statistics["accuracy"]["mean"],
        statistics["precision"]["mean"],
        statistics["recall"]["mean"],
        statistics["f1"]["mean"],
        statistics["iou"]["mean"],
    ]

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        names,
        values
    )

    plt.ylim(
        0,
        1
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Average Model Performance"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    # --------------------------------------------
    # Display values above bars
    # --------------------------------------------

    for bar, value in zip(
        bars,
        values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 0.02,

            f"{value:.4f}",

            ha="center"
        )

    plt.tight_layout()

    output = (
        PLOTS_DIR
        / "metrics_comparison.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# SAVE SAMPLE INFORMATION
# ============================================================

def save_sample_information(
    best,
    worst
):
    """
    Save information about best and worst predictions.
    """

    print()
    print("=" * 60)
    print("BEST PREDICTIONS")
    print("=" * 60)

    for index, row in enumerate(
        best,
        start=1
    ):

        print(
            f"{index}. "
            f"{row['filename']} "
            f"| F1={row['f1']:.4f} "
            f"| IoU={row['iou']:.4f}"
        )

        output = (
            SAMPLES_DIR
            / f"best_{index:02d}.txt"
        )

        with open(
            output,
            "w"
        ) as file:

            file.write(
                "BEST PREDICTION\n"
            )

            file.write(
                "=" * 50
                + "\n\n"
            )

            for key, value in row.items():

                file.write(
                    f"{key}: {value}\n"
                )

    print()
    print("=" * 60)
    print("WORST PREDICTIONS")
    print("=" * 60)

    for index, row in enumerate(
        worst,
        start=1
    ):

        print(
            f"{index}. "
            f"{row['filename']} "
            f"| F1={row['f1']:.4f} "
            f"| IoU={row['iou']:.4f}"
        )

        output = (
            SAMPLES_DIR
            / f"worst_{index:02d}.txt"
        )

        with open(
            output,
            "w"
        ) as file:

            file.write(
                "WORST PREDICTION\n"
            )

            file.write(
                "=" * 50
                + "\n\n"
            )

            for key, value in row.items():

                file.write(
                    f"{key}: {value}\n"
                )


# ============================================================
# WRITE TEXT REPORT
# ============================================================

def write_summary(
    results,
    statistics,
    confusion,
    pixel_metrics,
    best,
    worst,
):
    """
    Create a human-readable evaluation report.
    """

    with open(
        SUMMARY_FILE,
        "w"
    ) as file:

        file.write(
            "URBAN CHANGE DETECTION\n"
        )

        file.write(
            "AUTOMATED EVALUATION REPORT\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        # --------------------------------------------
        # Dataset
        # --------------------------------------------

        file.write(
            "TEST DATASET\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        file.write(
            f"Number of test images: "
            f"{len(results)}\n\n"
        )

        # --------------------------------------------
        # Macro statistics
        # --------------------------------------------

        file.write(
            "MACRO-AVERAGE METRICS\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "iou",
        ]:

            data = statistics[metric]

            file.write(
                f"\n{metric.upper()}\n"
            )

            file.write(
                f"Mean   : {data['mean']:.6f}\n"
            )

            file.write(
                f"Std    : {data['std']:.6f}\n"
            )

            file.write(
                f"Median : {data['median']:.6f}\n"
            )

            file.write(
                f"Min    : {data['minimum']:.6f}\n"
            )

            file.write(
                f"Max    : {data['maximum']:.6f}\n"
            )

        # --------------------------------------------
        # Pixel metrics
        # --------------------------------------------

        file.write(
            "\n\n"
            "AGGREGATED PIXEL-LEVEL METRICS\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        file.write(
            f"Accuracy  : "
            f"{pixel_metrics['accuracy']:.6f}\n"
        )

        file.write(
            f"Precision : "
            f"{pixel_metrics['precision']:.6f}\n"
        )

        file.write(
            f"Recall    : "
            f"{pixel_metrics['recall']:.6f}\n"
        )

        file.write(
            f"F1-score  : "
            f"{pixel_metrics['f1']:.6f}\n"
        )

        file.write(
            f"IoU       : "
            f"{pixel_metrics['iou']:.6f}\n"
        )

        # --------------------------------------------
        # Confusion matrix
        # --------------------------------------------

        file.write(
            "\n\n"
            "CONFUSION MATRIX\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        file.write(
            f"TP: {confusion['TP']}\n"
        )

        file.write(
            f"TN: {confusion['TN']}\n"
        )

        file.write(
            f"FP: {confusion['FP']}\n"
        )

        file.write(
            f"FN: {confusion['FN']}\n"
        )

        # --------------------------------------------
        # Best predictions
        # --------------------------------------------

        file.write(
            "\n\n"
            "TOP 5 PREDICTIONS\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        for index, row in enumerate(
            best,
            start=1
        ):

            file.write(
                f"{index}. "
                f"{row['filename']} | "
                f"F1={row['f1']:.6f} | "
                f"IoU={row['iou']:.6f}\n"
            )

        # --------------------------------------------
        # Worst predictions
        # --------------------------------------------

        file.write(
            "\n\n"
            "BOTTOM 5 PREDICTIONS\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        for index, row in enumerate(
            worst,
            start=1
        ):

            file.write(
                f"{index}. "
                f"{row['filename']} | "
                f"F1={row['f1']:.6f} | "
                f"IoU={row['iou']:.6f}\n"
            )

    print()
    print(
        f"Summary saved:\n{SUMMARY_FILE}"
    )


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

def save_json_summary(
    statistics,
    confusion,
    pixel_metrics,
):
    """
    Save machine-readable report.
    """

    report = {
        "dataset": "LEVIR-CD",
        "number_of_test_images": 128,

        "statistics": statistics,

        "aggregated_confusion_matrix":
            confusion,

        "aggregated_pixel_metrics":
            pixel_metrics,
    }

    with open(
        SUMMARY_JSON,
        "w"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        f"JSON summary saved:\n{SUMMARY_JSON}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("URBAN CHANGE DETECTION")
    print("AUTOMATED EVALUATION REPORT")
    print("=" * 60)
    print()

    # --------------------------------------------
    # Directories
    # --------------------------------------------

    create_directories()

    # --------------------------------------------
    # Check files
    # --------------------------------------------

    check_input_files()

    # --------------------------------------------
    # Load results
    # --------------------------------------------

    results = load_results()

    # --------------------------------------------
    # Calculate statistics
    # --------------------------------------------

    print("=" * 60)
    print("CALCULATING STATISTICS")
    print("=" * 60)

    statistics = calculate_statistics(
        results
    )

    confusion = calculate_confusion_matrix(
        results
    )

    pixel_metrics = calculate_pixel_metrics(
        confusion
    )

    best, worst = find_best_worst(
        results
    )

    print("Statistics calculated. ✅")
    print()

    # --------------------------------------------
    # Generate plots
    # --------------------------------------------

    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_f1_distribution(
        results
    )

    plot_iou_distribution(
        results
    )

    plot_precision_recall(
        results
    )

    plot_metric_comparison(
        statistics
    )

    print()

    # --------------------------------------------
    # Save samples
    # --------------------------------------------

    save_sample_information(
        best,
        worst
    )

    # --------------------------------------------
    # Write report
    # --------------------------------------------

    write_summary(
        results,
        statistics,
        confusion,
        pixel_metrics,
        best,
        worst,
    )

    save_json_summary(
        statistics,
        confusion,
        pixel_metrics,
    )

    # --------------------------------------------
    # Final output
    # --------------------------------------------

    print()
    print("=" * 60)
    print("EVALUATION REPORT COMPLETE ✅")
    print("=" * 60)

    print()
    print("Report directory:")
    print(REPORT_DIR)

    print()
    print("Summary:")
    print(SUMMARY_FILE)

    print()
    print("Plots:")
    print(PLOTS_DIR)

    print()
    print("Best/Worst samples:")
    print(SAMPLES_DIR)

    print()
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()