# ============================================================
# FILE:
# /Users/ahasanhabibkawsar/Documents/Urban_Change_Detection/
# threshold_sensitivity_analysis.py
#
# PURPOSE:
# Threshold Sensitivity Analysis for Urban Change Detection
# ============================================================

import os
import json
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# ============================================================
# YOUR ACTUAL PROJECT IMPORTS
# ============================================================

from models.siamese_unet import SiameseUNet
from dataset.levir_dataset import LEVIRCDDataset


# ============================================================
# START TIMER
# ============================================================

start_time = time.time()


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    "/Users/ahasanhabibkawsar/"
    "Documents/Urban_Change_Detection"
)


DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "LEVIR-CD"
)


TEST_A_DIR = os.path.join(
    DATASET_ROOT,
    "test",
    "A"
)


TEST_B_DIR = os.path.join(
    DATASET_ROOT,
    "test",
    "B"
)


TEST_LABEL_DIR = os.path.join(
    DATASET_ROOT,
    "test",
    "label"
)


CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "best_model.pth"
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "threshold_analysis"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# THRESHOLD SETTINGS
# ============================================================

CURRENT_THRESHOLD = 0.45

THRESHOLD_START = 0.10
THRESHOLD_END = 0.80
THRESHOLD_STEP = 0.025


THRESHOLDS = np.arange(
    THRESHOLD_START,
    THRESHOLD_END + 0.0001,
    THRESHOLD_STEP
)


# ============================================================
# PRINT HEADER
# ============================================================

print()
print("=" * 70)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 70)

print()
print(f"Device            : {DEVICE}")

print(
    f"Dataset root      : "
    f"{DATASET_ROOT}"
)

print(
    f"Test A            : "
    f"{TEST_A_DIR}"
)

print(
    f"Test B            : "
    f"{TEST_B_DIR}"
)

print(
    f"Test labels       : "
    f"{TEST_LABEL_DIR}"
)

print(
    f"Checkpoint        : "
    f"{CHECKPOINT_PATH}"
)

print(
    f"Current threshold : "
    f"{CURRENT_THRESHOLD}"
)

print(
    f"Threshold range   : "
    f"{THRESHOLD_START} → {THRESHOLD_END}"
)

print(
    f"Step              : "
    f"{THRESHOLD_STEP}"
)

print(
    f"Output directory  : "
    f"{OUTPUT_DIR}"
)

print()


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("=" * 70)
print("CHECKING REQUIRED FILES")
print("=" * 70)

required_paths = {
    "Dataset root": DATASET_ROOT,
    "Test A": TEST_A_DIR,
    "Test B": TEST_B_DIR,
    "Test labels": TEST_LABEL_DIR,
    "Checkpoint": CHECKPOINT_PATH,
}


for name, path in required_paths.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\n{name} not found:\n{path}"
        )


print()
print("All required files found. ✅")
print()


# ============================================================
# CREATE DATASET
# ============================================================

print("=" * 70)
print("CREATING TEST DATASET")
print("=" * 70)

test_dataset = LEVIRCDDataset(
    image_a_dir=TEST_A_DIR,
    image_b_dir=TEST_B_DIR,
    label_dir=TEST_LABEL_DIR,
    image_size=256,
    augment=False
)


print()
print(
    f"Valid test samples: "
    f"{len(test_dataset)}"
)


# ============================================================
# CREATE DATALOADER
# ============================================================

BATCH_SIZE = 4


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print(
    f"Test batches: "
    f"{len(test_loader)}"
)

print()


# ============================================================
# CREATE MODEL
# ============================================================

print("=" * 70)
print("CREATING SIAMESE U-NET")
print("=" * 70)

model = SiameseUNet()

model = model.to(DEVICE)


# ============================================================
# MODEL PARAMETERS
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print()

print(
    f"Total parameters    : "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

print()


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("=" * 70)
print("LOADING CHECKPOINT")
print("=" * 70)

print()
print(
    f"Checkpoint: "
    f"{CHECKPOINT_PATH}"
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)


# ============================================================
# HANDLE DIFFERENT CHECKPOINT FORMATS
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:

        # Sometimes the checkpoint itself
        # is a state dictionary.

        try:

            model.load_state_dict(
                checkpoint
            )

        except Exception as error:

            raise RuntimeError(
                "Unable to load model checkpoint.\n"
                f"Checkpoint keys: "
                f"{list(checkpoint.keys())}\n"
                f"Original error: {error}"
            )

else:

    model.load_state_dict(
        checkpoint
    )


# ============================================================
# CHECKPOINT INFORMATION
# ============================================================

if isinstance(checkpoint, dict):

    if "epoch" in checkpoint:

        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if "val_f1" in checkpoint:

        print(
            f"Validation F1: "
            f"{checkpoint['val_f1']}"
        )

    elif "validation_f1" in checkpoint:

        print(
            f"Validation F1: "
            f"{checkpoint['validation_f1']}"
        )

    else:

        print(
            "Validation F1: Unknown"
        )


print()
print(
    "Model loaded successfully. ✅"
)


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()

print(
    "Model switched to evaluation mode. ✅"
)

print()


# ============================================================
# COLLECT PROBABILITIES
# ============================================================

print("=" * 70)
print("COLLECTING MODEL PROBABILITIES")
print("=" * 70)

all_probabilities = []
all_targets = []


with torch.no_grad():

    for batch_index, batch in enumerate(
        test_loader
    ):

        # ====================================================
        # DATASET RETURNS DICTIONARY
        # ====================================================

        if not isinstance(batch, dict):

            raise TypeError(
                "Expected DataLoader batch "
                "to be a dictionary."
            )


        # ====================================================
        # GET TENSORS
        # ====================================================

        before = batch["before"]
        after = batch["after"]
        mask = batch["mask"]


        # ====================================================
        # MOVE TO DEVICE
        # ====================================================

        before = before.to(
            DEVICE,
            non_blocking=False
        )

        after = after.to(
            DEVICE,
            non_blocking=False
        )

        mask = mask.to(
            DEVICE,
            non_blocking=False
        )


        # ====================================================
        # FORWARD PASS
        # ====================================================

        logits = model(
            before,
            after
        )


        # ====================================================
        # SIGMOID
        # ====================================================

        probabilities = torch.sigmoid(
            logits
        )


        # ====================================================
        # SAVE CPU ARRAYS
        # ====================================================

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_targets.append(
            mask.cpu().numpy()
        )


        print(
            f"Processed batch "
            f"{batch_index + 1}/"
            f"{len(test_loader)}"
        )


# ============================================================
# COMBINE RESULTS
# ============================================================

all_probabilities = np.concatenate(
    all_probabilities,
    axis=0
)


all_targets = np.concatenate(
    all_targets,
    axis=0
)


print()

print(
    f"Probabilities: "
    f"{all_probabilities.shape}"
)

print(
    f"Targets      : "
    f"{all_targets.shape}"
)

print()


# ============================================================
# FLATTEN
# ============================================================

probabilities = (
    all_probabilities.reshape(-1)
)


targets = (
    all_targets.reshape(-1)
)


# ============================================================
# CONVERT TARGETS TO BINARY
# ============================================================

targets = (
    targets >= 0.5
).astype(np.uint8)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    probabilities,
    targets,
    threshold
):

    # --------------------------------------------------------
    # Binary prediction
    # --------------------------------------------------------

    predictions = (
        probabilities >= threshold
    ).astype(np.uint8)


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tp = np.sum(
        (predictions == 1) &
        (targets == 1)
    )


    tn = np.sum(
        (predictions == 0) &
        (targets == 0)
    )


    fp = np.sum(
        (predictions == 1) &
        (targets == 0)
    )


    fn = np.sum(
        (predictions == 0) &
        (targets == 1)
    )


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    total = (
        tp +
        tn +
        fp +
        fn
    )


    if total > 0:

        accuracy = (
            (tp + tn) /
            total
        )

    else:

        accuracy = 0.0


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if (tp + fp) > 0:

        precision = (
            tp /
            (tp + fp)
        )

    else:

        precision = 0.0


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if (tp + fn) > 0:

        recall = (
            tp /
            (tp + fn)
        )

    else:

        recall = 0.0


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if (precision + recall) > 0:

        f1 = (
            2 *
            precision *
            recall /
            (precision + recall)
        )

    else:

        f1 = 0.0


    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    if (tp + fp + fn) > 0:

        iou = (
            tp /
            (tp + fp + fn)
        )

    else:

        iou = 0.0


    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


# ============================================================
# RUN ALL THRESHOLDS
# ============================================================

print("=" * 70)
print("RUNNING THRESHOLD ANALYSIS")
print("=" * 70)

print()

results = []


for threshold in THRESHOLDS:

    metrics = calculate_metrics(
        probabilities,
        targets,
        threshold
    )

    results.append(
        metrics
    )


    print(
        f"{threshold:.3f}    "
        f"{metrics['precision']:.4f}    "
        f"{metrics['recall']:.4f}    "
        f"{metrics['f1']:.4f}    "
        f"{metrics['iou']:.4f}"
    )


# ============================================================
# FIND CURRENT THRESHOLD RESULT
# ============================================================

current_result = min(
    results,
    key=lambda result:
    abs(
        result["threshold"] -
        CURRENT_THRESHOLD
    )
)


# ============================================================
# FIND BEST F1
# ============================================================

best_f1_result = max(
    results,
    key=lambda result:
    result["f1"]
)


# ============================================================
# FIND BEST IoU
# ============================================================

best_iou_result = max(
    results,
    key=lambda result:
    result["iou"]
)


# ============================================================
# PRINT BEST RESULTS
# ============================================================

print()

print("=" * 70)
print("BEST THRESHOLD RESULTS")
print("=" * 70)

print()

print(
    "Current threshold:"
)

print(
    f"Threshold : "
    f"{current_result['threshold']:.3f}"
)

print(
    f"Precision : "
    f"{current_result['precision']:.6f}"
)

print(
    f"Recall    : "
    f"{current_result['recall']:.6f}"
)

print(
    f"F1        : "
    f"{current_result['f1']:.6f}"
)

print(
    f"IoU       : "
    f"{current_result['iou']:.6f}"
)

print()

print(
    "Best F1 threshold:"
)

print(
    f"Threshold : "
    f"{best_f1_result['threshold']:.3f}"
)

print(
    f"F1        : "
    f"{best_f1_result['f1']:.6f}"
)

print(
    f"Precision : "
    f"{best_f1_result['precision']:.6f}"
)

print(
    f"Recall    : "
    f"{best_f1_result['recall']:.6f}"
)

print(
    f"IoU       : "
    f"{best_f1_result['iou']:.6f}"
)

print()

print(
    "Best IoU threshold:"
)

print(
    f"Threshold : "
    f"{best_iou_result['threshold']:.3f}"
)

print(
    f"IoU       : "
    f"{best_iou_result['iou']:.6f}"
)

print(
    f"F1        : "
    f"{best_iou_result['f1']:.6f}"
)

print()


# ============================================================
# SAVE JSON
# ============================================================

json_path = os.path.join(
    OUTPUT_DIR,
    "threshold_results.json"
)


json_data = {
    "current_threshold": CURRENT_THRESHOLD,

    "threshold_range": {
        "start": THRESHOLD_START,
        "end": THRESHOLD_END,
        "step": THRESHOLD_STEP
    },

    "current_result": current_result,

    "best_f1": best_f1_result,

    "best_iou": best_iou_result,

    "all_results": results
}


with open(
    json_path,
    "w"
) as file:

    json.dump(
        json_data,
        file,
        indent=4
    )


print(
    f"Results saved:\n"
    f"{json_path}"
)


# ============================================================
# PREPARE PLOT DATA
# ============================================================

threshold_values = [
    result["threshold"]
    for result in results
]


precision_values = [
    result["precision"]
    for result in results
]


recall_values = [
    result["recall"]
    for result in results
]


f1_values = [
    result["f1"]
    for result in results
]


iou_values = [
    result["iou"]
    for result in results
]


# ============================================================
# F1 VS THRESHOLD
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    threshold_values,
    f1_values,
    marker="o",
    label="F1"
)

plt.axvline(
    CURRENT_THRESHOLD,
    linestyle="--",
    label=(
        f"Current = "
        f"{CURRENT_THRESHOLD:.2f}"
    )
)

plt.axvline(
    best_f1_result["threshold"],
    linestyle=":",
    label=(
        f"Best F1 = "
        f"{best_f1_result['threshold']:.3f}"
    )
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "F1 Score"
)

plt.title(
    "F1 Score vs Classification Threshold"
)

plt.grid(True)

plt.legend()

plt.tight_layout()


f1_plot_path = os.path.join(
    OUTPUT_DIR,
    "f1_vs_threshold.png"
)


plt.savefig(
    f1_plot_path,
    dpi=300
)

plt.close()


print(
    f"Saved: {f1_plot_path}"
)


# ============================================================
# IoU VS THRESHOLD
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    threshold_values,
    iou_values,
    marker="o",
    label="IoU"
)

plt.axvline(
    CURRENT_THRESHOLD,
    linestyle="--",
    label=(
        f"Current = "
        f"{CURRENT_THRESHOLD:.2f}"
    )
)

plt.axvline(
    best_iou_result["threshold"],
    linestyle=":",
    label=(
        f"Best IoU = "
        f"{best_iou_result['threshold']:.3f}"
    )
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "IoU"
)

plt.title(
    "IoU vs Classification Threshold"
)

plt.grid(True)

plt.legend()

plt.tight_layout()


iou_plot_path = os.path.join(
    OUTPUT_DIR,
    "iou_vs_threshold.png"
)


plt.savefig(
    iou_plot_path,
    dpi=300
)

plt.close()


print(
    f"Saved: {iou_plot_path}"
)


# ============================================================
# PRECISION / RECALL VS THRESHOLD
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    threshold_values,
    precision_values,
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_values,
    recall_values,
    marker="o",
    label="Recall"
)

plt.axvline(
    CURRENT_THRESHOLD,
    linestyle="--",
    label=(
        f"Current = "
        f"{CURRENT_THRESHOLD:.2f}"
    )
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Precision and Recall vs Classification Threshold"
)

plt.grid(True)

plt.legend()

plt.tight_layout()


precision_recall_plot_path = os.path.join(
    OUTPUT_DIR,
    "precision_recall_vs_threshold.png"
)


plt.savefig(
    precision_recall_plot_path,
    dpi=300
)

plt.close()


print(
    f"Saved: "
    f"{precision_recall_plot_path}"
)


# ============================================================
# F1 + IoU COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    threshold_values,
    f1_values,
    marker="o",
    label="F1"
)

plt.plot(
    threshold_values,
    iou_values,
    marker="o",
    label="IoU"
)

plt.axvline(
    CURRENT_THRESHOLD,
    linestyle="--",
    label=(
        f"Current = "
        f"{CURRENT_THRESHOLD:.2f}"
    )
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "F1 and IoU vs Classification Threshold"
)

plt.grid(True)

plt.legend()

plt.tight_layout()


comparison_plot_path = os.path.join(
    OUTPUT_DIR,
    "f1_iou_comparison.png"
)


plt.savefig(
    comparison_plot_path,
    dpi=300
)

plt.close()


print(
    f"Saved: "
    f"{comparison_plot_path}"
)


# ============================================================
# WRITE TEXT REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "threshold_report.txt"
)


with open(
    report_path,
    "w"
) as report:

    report.write(
        "URBAN CHANGE DETECTION\n"
    )

    report.write(
        "THRESHOLD SENSITIVITY ANALYSIS\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )

    report.write(
        f"Device: {DEVICE}\n"
    )

    report.write(
        f"Dataset root: {DATASET_ROOT}\n"
    )

    report.write(
        f"Checkpoint: {CHECKPOINT_PATH}\n"
    )

    report.write(
        f"Test samples: {len(test_dataset)}\n"
    )

    report.write(
        f"Current threshold: "
        f"{CURRENT_THRESHOLD:.3f}\n\n"
    )


    report.write(
        "CURRENT THRESHOLD\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        f"Threshold : "
        f"{current_result['threshold']:.3f}\n"
    )

    report.write(
        f"Accuracy  : "
        f"{current_result['accuracy']:.6f}\n"
    )

    report.write(
        f"Precision : "
        f"{current_result['precision']:.6f}\n"
    )

    report.write(
        f"Recall    : "
        f"{current_result['recall']:.6f}\n"
    )

    report.write(
        f"F1        : "
        f"{current_result['f1']:.6f}\n"
    )

    report.write(
        f"IoU       : "
        f"{current_result['iou']:.6f}\n\n"
    )


    report.write(
        "BEST F1 THRESHOLD\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        f"Threshold : "
        f"{best_f1_result['threshold']:.3f}\n"
    )

    report.write(
        f"Accuracy  : "
        f"{best_f1_result['accuracy']:.6f}\n"
    )

    report.write(
        f"Precision : "
        f"{best_f1_result['precision']:.6f}\n"
    )

    report.write(
        f"Recall    : "
        f"{best_f1_result['recall']:.6f}\n"
    )

    report.write(
        f"F1        : "
        f"{best_f1_result['f1']:.6f}\n"
    )

    report.write(
        f"IoU       : "
        f"{best_f1_result['iou']:.6f}\n\n"
    )


    report.write(
        "BEST IoU THRESHOLD\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        f"Threshold : "
        f"{best_iou_result['threshold']:.3f}\n"
    )

    report.write(
        f"Accuracy  : "
        f"{best_iou_result['accuracy']:.6f}\n"
    )

    report.write(
        f"Precision : "
        f"{best_iou_result['precision']:.6f}\n"
    )

    report.write(
        f"Recall    : "
        f"{best_iou_result['recall']:.6f}\n"
    )

    report.write(
        f"F1        : "
        f"{best_iou_result['f1']:.6f}\n"
    )

    report.write(
        f"IoU       : "
        f"{best_iou_result['iou']:.6f}\n\n"
    )


    report.write(
        "ALL THRESHOLD RESULTS\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        "Threshold | Precision | Recall | "
        "F1 | IoU\n"
    )

    report.write(
        "-" * 70 + "\n"
    )


    for result in results:

        report.write(
            f"{result['threshold']:.3f} | "
            f"{result['precision']:.6f} | "
            f"{result['recall']:.6f} | "
            f"{result['f1']:.6f} | "
            f"{result['iou']:.6f}\n"
        )


print(
    f"Report saved:\n"
    f"{report_path}"
)


# ============================================================
# RUNTIME
# ============================================================

runtime = (
    time.time() -
    start_time
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()

print("=" * 70)
print("THRESHOLD SENSITIVITY ANALYSIS COMPLETE 🎉🔥")
print("=" * 70)

print()

print(
    f"Images analyzed : "
    f"{len(test_dataset)}"
)

print(
    f"Thresholds tested: "
    f"{len(results)}"
)

print()

print(
    f"Current threshold : "
    f"{CURRENT_THRESHOLD:.3f}"
)

print(
    f"Current F1        : "
    f"{current_result['f1']:.6f}"
)

print(
    f"Current IoU       : "
    f"{current_result['iou']:.6f}"
)

print()

print(
    f"Best F1 threshold : "
    f"{best_f1_result['threshold']:.3f}"
)

print(
    f"Best F1           : "
    f"{best_f1_result['f1']:.6f}"
)

print()

print(
    f"Best IoU threshold: "
    f"{best_iou_result['threshold']:.3f}"
)

print(
    f"Best IoU          : "
    f"{best_iou_result['iou']:.6f}"
)

print()

print(
    f"Runtime           : "
    f"{runtime:.2f} seconds"
)

print()

print(
    f"Results directory:\n"
    f"{OUTPUT_DIR}"
)

print()

print(
    "Everything completed successfully. 🎉🔥"
)