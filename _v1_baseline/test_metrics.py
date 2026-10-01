"""
Test change-detection metrics.
"""

import torch

from training.metrics import (
    calculate_metrics,
)


def main():

    print("=" * 60)
    print("METRICS TEST")
    print("=" * 60)

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    if torch.backends.mps.is_available():

        device = torch.device("mps")

    elif torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print()
    print("Device:", device)

    # -------------------------------------------------
    # Create simple test data
    # -------------------------------------------------

    targets = torch.tensor(
        [
            [
                [
                    [1, 1],
                    [0, 0],
                ]
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    # Very confident correct prediction

    logits = torch.tensor(
        [
            [
                [
                    [5.0, 5.0],
                    [-5.0, -5.0],
                ]
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    # -------------------------------------------------
    # Calculate metrics
    # -------------------------------------------------

    metrics = calculate_metrics(
        logits,
        targets,
    )

    print()
    print("Metrics:")
    print("-" * 40)

    print(
        "Accuracy :",
        metrics["accuracy"],
    )

    print(
        "Precision:",
        metrics["precision"],
    )

    print(
        "Recall   :",
        metrics["recall"],
    )

    print(
        "F1       :",
        metrics["f1"],
    )

    print(
        "IoU      :",
        metrics["iou"],
    )

    print()
    print(
        "TP:",
        metrics["tp"],
        "TN:",
        metrics["tn"],
        "FP:",
        metrics["fp"],
        "FN:",
        metrics["fn"],
    )

    # -------------------------------------------------
    # Verify perfect prediction
    # -------------------------------------------------

    if (
        metrics["precision"] == 1.0
        and metrics["recall"] == 1.0
        and metrics["f1"] == 1.0
        and metrics["iou"] == 1.0
    ):

        print()
        print(
            "Perfect prediction test: PASSED ✅"
        )

    else:

        raise RuntimeError(
            "Metrics calculation failed."
        )

    print()
    print("=" * 60)
    print("METRICS TEST: PASSED")
    print("=" * 60)


if __name__ == "__main__":

    main()