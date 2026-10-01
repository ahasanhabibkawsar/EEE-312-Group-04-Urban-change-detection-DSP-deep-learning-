"""
============================================================
EVALUATION REPORT
============================================================
"""

import json
import os


# ============================================================
# SAVE TEXT REPORT
# ============================================================

def save_text_report(
    metrics,
    output_path,
    test_samples=128
):

    with open(
        output_path,
        "w"
    ) as file:

        file.write(
            "============================================================\n"
        )

        file.write(
            "URBAN CHANGE DETECTION\n"
        )

        file.write(
            "FINAL TEST EVALUATION REPORT\n"
        )

        file.write(
            "============================================================\n\n"
        )

        file.write(
            f"Test samples : {test_samples}\n\n"
        )

        file.write(
            "Performance Metrics\n"
        )

        file.write(
            "------------------------------------------------------------\n"
        )

        file.write(
            f"Accuracy  : {metrics['accuracy']:.6f}\n"
        )

        file.write(
            f"Precision : {metrics['precision']:.6f}\n"
        )

        file.write(
            f"Recall    : {metrics['recall']:.6f}\n"
        )

        file.write(
            f"F1-score  : {metrics['f1']:.6f}\n"
        )

        file.write(
            f"IoU       : {metrics['iou']:.6f}\n"
        )

        file.write(
            "\nConfusion Matrix\n"
        )

        file.write(
            "------------------------------------------------------------\n"
        )

        file.write(
            f"TP : {metrics['tp']:,}\n"
        )

        file.write(
            f"TN : {metrics['tn']:,}\n"
        )

        file.write(
            f"FP : {metrics['fp']:,}\n"
        )

        file.write(
            f"FN : {metrics['fn']:,}\n"
        )

        file.write(
            "\n============================================================\n"
        )


# ============================================================
# SAVE JSON REPORT
# ============================================================

def save_json_report(
    metrics,
    output_path,
    test_samples=128
):

    report = {
        "dataset": "LEVIR-CD",
        "test_samples": test_samples,
        "metrics": metrics
    }

    with open(
        output_path,
        "w"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )