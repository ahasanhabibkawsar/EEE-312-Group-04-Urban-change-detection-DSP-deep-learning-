"""
============================================================
URBAN CHANGE DETECTION
THRESHOLD ANALYSIS
============================================================

Purpose:
    Find the optimal probability threshold for binary
    change detection using the validation set.

Important:
    Threshold selection is performed ONLY on validation data.

    The test set remains untouched until final evaluation.

============================================================
"""

from pathlib import Path
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset.levir_dataset import LEVIRCDDataset
from models.siamese_unet import SiameseUNet


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "data" / "LEVIR-CD"

VAL_A = DATASET_ROOT / "val" / "A"
VAL_B = DATASET_ROOT / "val" / "B"
VAL_LABEL = DATASET_ROOT / "val" / "label"

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "threshold_analysis_results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

def create_validation_dataset():

    print("=" * 60)
    print("CREATING VALIDATION DATASET")
    print("=" * 60)

    dataset = LEVIRCDDataset(
        image_a_dir=VAL_A,
        image_b_dir=VAL_B,
        label_dir=VAL_LABEL,
        image_size=256,
        augment=False
    )

    print(
        f"Validation samples: {len(dataset)}"
    )

    return dataset


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_dataloader(dataset):

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(
        f"Validation batches: {len(loader)}"
    )

    return loader


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(device):

    print()
    print("=" * 60)
    print("CREATING SIAMESE U-NET")
    print("=" * 60)

    model = SiameseUNet(
        in_channels=4
    )

    model = model.to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters    : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model, device):

    print()
    print("=" * 60)
    print("LOADING BEST MODEL")
    print("=" * 60)

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    # --------------------------------------------------------
    # Handle checkpoint format
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            epoch = checkpoint.get(
                "epoch",
                "Unknown"
            )

            validation_f1 = checkpoint.get(
                "best_val_f1",
                checkpoint.get(
                    "val_f1",
                    "Unknown"
                )
            )

            print(
                f"Checkpoint epoch: "
                f"{epoch}"
            )

            print(
                f"Validation F1: "
                f"{validation_f1}"
            )

        elif "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    print(
        "Model loaded successfully. ✅"
    )

    return model


# ============================================================
# COLLECT VALIDATION OUTPUTS
# ============================================================

def collect_predictions(
    model,
    loader,
    device
):

    print()
    print("=" * 60)
    print("RUNNING VALIDATION INFERENCE")
    print("=" * 60)

    all_probabilities = []
    all_targets = []

    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader,
            start=1
        ):

            before = batch[
                "before"
            ].to(device)

            after = batch[
                "after"
            ].to(device)

            mask = batch[
                "mask"
            ].to(device)

            logits = model(
                before,
                after
            )

            probabilities = torch.sigmoid(
                logits
            )

            all_probabilities.append(
                probabilities.detach().cpu()
            )

            all_targets.append(
                mask.detach().cpu()
            )

            print(
                f"Processed batch "
                f"{batch_index}/{len(loader)}"
            )

    probabilities = torch.cat(
        all_probabilities,
        dim=0
    )

    targets = torch.cat(
        all_targets,
        dim=0
    )

    print()
    print(
        f"Probabilities: "
        f"{probabilities.shape}"
    )

    print(
        f"Targets      : "
        f"{targets.shape}"
    )

    return probabilities, targets


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    probabilities,
    targets,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).float()

    targets = (
        targets >= 0.5
    ).float()

    tp = (
        (predictions == 1)
        & (targets == 1)
    ).sum().item()

    tn = (
        (predictions == 0)
        & (targets == 0)
    ).sum().item()

    fp = (
        (predictions == 1)
        & (targets == 0)
    ).sum().item()

    fn = (
        (predictions == 0)
        & (targets == 1)
    ).sum().item()

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    total = (
        tp + tn + fp + fn
    )

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0.0
    )

    return {
        "threshold": threshold,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "iou": iou,

        "tp": tp,

        "tn": tn,

        "fp": fp,

        "fn": fn,
    }


# ============================================================
# TEST ALL THRESHOLDS
# ============================================================

def evaluate_thresholds(
    probabilities,
    targets
):

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    results = []

    print()
    print("=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)

    print()

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'IoU':<12}"
    )

    print("-" * 60)

    for threshold in thresholds:

        result = calculate_metrics(
            probabilities,
            targets,
            float(threshold)
        )

        results.append(
            result
        )

        print(
            f"{threshold:<12.2f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1']:<12.4f}"
            f"{result['iou']:<12.4f}"
        )

    return results


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

def find_best_threshold(
    results
):

    best = max(
        results,
        key=lambda item: item["f1"]
    )

    return best


# ============================================================
# SAVE JSON
# ============================================================

def save_results(
    results,
    best
):

    output = (
        RESULTS_DIR
        / "threshold_results.json"
    )

    data = {
        "dataset": "LEVIR-CD",

        "purpose":
            "Validation threshold selection",

        "best_threshold":
            best,

        "all_thresholds":
            results,
    }

    with open(
        output,
        "w"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )

    print()
    print(
        f"Results saved:\n{output}"
    )


# ============================================================
# PLOT F1
# ============================================================

def plot_f1(results):

    thresholds = [
        item["threshold"]
        for item in results
    ]

    f1_values = [
        item["f1"]
        for item in results
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        thresholds,
        f1_values,
        marker="o"
    )

    plt.xlabel(
        "Probability Threshold"
    )

    plt.ylabel(
        "F1-score"
    )

    plt.title(
        "F1-score vs Probability Threshold"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        RESULTS_DIR
        / "f1_vs_threshold.png"
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
# PLOT IOU
# ============================================================

def plot_iou(results):

    thresholds = [
        item["threshold"]
        for item in results
    ]

    iou_values = [
        item["iou"]
        for item in results
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        thresholds,
        iou_values,
        marker="o"
    )

    plt.xlabel(
        "Probability Threshold"
    )

    plt.ylabel(
        "IoU"
    )

    plt.title(
        "IoU vs Probability Threshold"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        RESULTS_DIR
        / "iou_vs_threshold.png"
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
# PLOT PRECISION / RECALL
# ============================================================

def plot_precision_recall(
    results
):

    thresholds = [
        item["threshold"]
        for item in results
    ]

    precision = [
        item["precision"]
        for item in results
    ]

    recall = [
        item["recall"]
        for item in results
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        thresholds,
        precision,
        marker="o",
        label="Precision"
    )

    plt.plot(
        thresholds,
        recall,
        marker="o",
        label="Recall"
    )

    plt.xlabel(
        "Probability Threshold"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Precision and Recall vs Threshold"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        RESULTS_DIR
        / "precision_recall_vs_threshold.png"
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
# WRITE TEXT REPORT
# ============================================================

def write_report(
    results,
    best
):

    output = (
        RESULTS_DIR
        / "threshold_report.txt"
    )

    with open(
        output,
        "w"
    ) as file:

        file.write(
            "URBAN CHANGE DETECTION\n"
        )

        file.write(
            "THRESHOLD ANALYSIS REPORT\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            "Dataset: LEVIR-CD\n"
        )

        file.write(
            "Split: Validation\n\n"
        )

        file.write(
            "BEST THRESHOLD\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        file.write(
            f"Threshold : "
            f"{best['threshold']:.2f}\n"
        )

        file.write(
            f"Precision : "
            f"{best['precision']:.6f}\n"
        )

        file.write(
            f"Recall    : "
            f"{best['recall']:.6f}\n"
        )

        file.write(
            f"F1-score  : "
            f"{best['f1']:.6f}\n"
        )

        file.write(
            f"IoU       : "
            f"{best['iou']:.6f}\n"
        )

        file.write(
            "\n\n"
        )

        file.write(
            "ALL THRESHOLDS\n"
        )

        file.write(
            "-" * 60
            + "\n\n"
        )

        for item in results:

            file.write(
                f"Threshold: "
                f"{item['threshold']:.2f}\n"
            )

            file.write(
                f"Precision: "
                f"{item['precision']:.6f}\n"
            )

            file.write(
                f"Recall: "
                f"{item['recall']:.6f}\n"
            )

            file.write(
                f"F1: "
                f"{item['f1']:.6f}\n"
            )

            file.write(
                f"IoU: "
                f"{item['iou']:.6f}\n"
            )

            file.write(
                "\n"
            )

    print(
        f"Report saved:\n{output}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("URBAN CHANGE DETECTION")
    print("THRESHOLD ANALYSIS")
    print("=" * 60)

    device = get_device()

    print()
    print(
        f"Device: {device}"
    )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = create_validation_dataset()

    loader = create_dataloader(
        dataset
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        device
    )

    model = load_checkpoint(
        model,
        device
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    probabilities, targets = (
        collect_predictions(
            model,
            loader,
            device
        )
    )

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    results = evaluate_thresholds(
        probabilities,
        targets
    )

    # --------------------------------------------------------
    # Best
    # --------------------------------------------------------

    best = find_best_threshold(
        results
    )

    print()
    print("=" * 60)
    print("BEST VALIDATION THRESHOLD")
    print("=" * 60)

    print(
        f"Threshold : "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{best['recall']:.6f}"
    )

    print(
        f"F1-score  : "
        f"{best['f1']:.6f}"
    )

    print(
        f"IoU       : "
        f"{best['iou']:.6f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        best
    )

    plot_f1(
        results
    )

    plot_iou(
        results
    )

    plot_precision_recall(
        results
    )

    write_report(
        results,
        best
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("THRESHOLD ANALYSIS COMPLETE ✅")
    print("=" * 60)

    print()
    print(
        "Results directory:"
    )

    print(
        RESULTS_DIR
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()