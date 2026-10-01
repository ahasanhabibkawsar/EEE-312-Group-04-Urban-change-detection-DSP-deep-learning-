"""
============================================================
URBAN CHANGE DETECTION
FALSE POSITIVE / FALSE NEGATIVE ERROR ANALYSIS
============================================================

Analyzes:
    - False Positives (FP)
    - False Negatives (FN)
    - True Positives (TP)
    - True Negatives (TN)

Generates:
    - Per-image error statistics
    - Overall error statistics
    - FP/FN distributions
    - FP/FN versus change area plots
    - Worst FP samples
    - Worst FN samples
    - Visualization of selected errors
    - Text report

Dataset:
    LEVIR-CD

Model:
    Siamese U-Net

Device:
    Apple Silicon MPS / CPU / CUDA

Threshold:
    0.45
"""

import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset.levir_dataset import LEVIRCDDataset
from models.siamese_unet import SiameseUNet
from config import load_best_threshold
from preprocessing.dsp_processing import denormalize_for_display


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "data" / "LEVIR-CD"

TEST_A_DIR = DATASET_ROOT / "test" / "A"
TEST_B_DIR = DATASET_ROOT / "test" / "B"
TEST_LABEL_DIR = DATASET_ROOT / "test" / "label"

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

OUTPUT_DIR = PROJECT_ROOT / "error_type_analysis"

WORST_FP_DIR = OUTPUT_DIR / "highest_fp"
WORST_FN_DIR = OUTPUT_DIR / "highest_fn"

IMAGE_SIZE = 256
BATCH_SIZE = 4

THRESHOLD = load_best_threshold()  # v2: threshold selected on the VALIDATION set

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    elif torch.cuda.is_available():

        return torch.device("cuda")

    else:

        return torch.device("cpu")


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    WORST_FP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    WORST_FN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def check_required_files():

    print()
    print("=" * 60)
    print("CHECKING REQUIRED FILES")
    print("=" * 60)

    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Test A       : {TEST_A_DIR}")
    print(f"Test B       : {TEST_B_DIR}")
    print(f"Test labels  : {TEST_LABEL_DIR}")
    print(f"Checkpoint   : {CHECKPOINT_PATH}")

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"\nDataset root not found:\n{DATASET_ROOT}"
        )

    if not TEST_A_DIR.exists():

        raise FileNotFoundError(
            f"\nTest A directory not found:\n{TEST_A_DIR}"
        )

    if not TEST_B_DIR.exists():

        raise FileNotFoundError(
            f"\nTest B directory not found:\n{TEST_B_DIR}"
        )

    if not TEST_LABEL_DIR.exists():

        raise FileNotFoundError(
            f"\nTest label directory not found:\n{TEST_LABEL_DIR}"
        )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{CHECKPOINT_PATH}"
        )

    print()
    print("All required files found. ✅")


# ============================================================
# CREATE DATASET
# ============================================================

def create_test_dataset():

    print()
    print("=" * 60)
    print("CREATING TEST DATASET")
    print("=" * 60)

    dataset = LEVIRCDDataset(

        image_a_dir=TEST_A_DIR,

        image_b_dir=TEST_B_DIR,

        label_dir=TEST_LABEL_DIR,

        image_size=IMAGE_SIZE,

        augment=False
    )

    print()
    print(f"Valid test samples: {len(dataset)}")

    return dataset


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_test_loader(dataset):

    print()
    print("=" * 60)
    print("CREATING TEST DATALOADER")
    print("=" * 60)

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=False
    )

    print()
    print(f"Test batches: {len(loader)}")

    return loader


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(device):

    print()
    print("=" * 60)
    print("CREATING SIAMESE U-NET")
    print("=" * 60)

    model = SiameseUNet()

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Total parameters    : {total_parameters:,}"
    )

    print(
        f"Trainable parameters: {trainable_parameters:,}"
    )

    model = model.to(device)

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model, device):

    print()
    print("=" * 60)
    print("LOADING CHECKPOINT")
    print("=" * 60)

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    checkpoint = torch.load(

        CHECKPOINT_PATH,

        map_location=device
    )

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

            model.load_state_dict(
                checkpoint
            )

        epoch = checkpoint.get(
            "epoch",
            "Unknown"
        )

        print(
            f"Checkpoint epoch: {epoch}"
        )

    else:

        model.load_state_dict(
            checkpoint
        )

        print(
            "Checkpoint format: state_dict"
        )

    print(
        "Model loaded successfully. ✅"
    )

    model.eval()

    print(
        "Model switched to evaluation mode. ✅"
    )


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_divide(a, b):

    if b == 0:

        return 0.0

    return float(a / b)


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(

    prediction,
    target
):

    prediction = prediction.astype(bool)

    target = target.astype(bool)

    tp = np.logical_and(
        prediction,
        target
    ).sum()

    tn = np.logical_and(
        ~prediction,
        ~target
    ).sum()

    fp = np.logical_and(
        prediction,
        ~target
    ).sum()

    fn = np.logical_and(
        ~prediction,
        target
    ).sum()

    precision = safe_divide(
        tp,
        tp + fp
    )

    recall = safe_divide(
        tp,
        tp + fn
    )

    f1 = safe_divide(

        2 * precision * recall,

        precision + recall
    )

    iou = safe_divide(

        tp,

        tp + fp + fn
    )

    accuracy = safe_divide(

        tp + tn,

        tp + tn + fp + fn
    )

    return {

        "tp": int(tp),

        "tn": int(tn),

        "fp": int(fp),

        "fn": int(fn),

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "iou": iou,

        "accuracy": accuracy
    }


# ============================================================
# SAVE ERROR VISUALIZATION
# ============================================================

def save_error_visualization(

    before,

    after,

    target,

    prediction,

    probability,

    filename,

    output_path
):

    # --------------------------------------------------------
    # Convert image
    # --------------------------------------------------------

    before = before.detach().cpu().numpy()

    after = after.detach().cpu().numpy()

    target = target.detach().cpu().numpy()

    prediction = prediction.detach().cpu().numpy()

    probability = probability.detach().cpu().numpy()

    # --------------------------------------------------------
    # CHW -> HWC
    # --------------------------------------------------------

    before = np.transpose(
        before,
        (1, 2, 0)
    )

    after = np.transpose(
        after,
        (1, 2, 0)
    )

    # --------------------------------------------------------
    # Display first 3 channels
    # --------------------------------------------------------

    if before.shape[-1] >= 3:

        before_display = denormalize_for_display(before)

        after_display = denormalize_for_display(after)

    else:

        before_display = before[:, :, 0]

        after_display = after[:, :, 0]

    # --------------------------------------------------------
    # Normalize RGB images
    # --------------------------------------------------------

    def normalize_image(image):

        image = image.astype(
            np.float32
        )

        min_value = image.min()

        max_value = image.max()

        if max_value - min_value > 1e-8:

            image = (
                image - min_value
            ) / (
                max_value - min_value
            )

        return image

    before_display = normalize_image(
        before_display
    )

    after_display = normalize_image(
        after_display
    )

    # --------------------------------------------------------
    # Remove dimensions
    # --------------------------------------------------------

    target = np.squeeze(target)

    prediction = np.squeeze(prediction)

    probability = np.squeeze(probability)

    # --------------------------------------------------------
    # Error maps
    # --------------------------------------------------------

    fp = np.logical_and(

        prediction == 1,

        target == 0
    )

    fn = np.logical_and(

        prediction == 0,

        target == 1
    )

    tp = np.logical_and(

        prediction == 1,

        target == 1
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(18, 10)
    )

    ax1 = plt.subplot(2, 4, 1)

    if before_display.ndim == 2:

        ax1.imshow(
            before_display,
            cmap="gray"
        )

    else:

        ax1.imshow(
            before_display
        )

    ax1.set_title(
        "Before"
    )

    ax1.axis("off")

    # --------------------------------------------------------

    ax2 = plt.subplot(2, 4, 2)

    if after_display.ndim == 2:

        ax2.imshow(
            after_display,
            cmap="gray"
        )

    else:

        ax2.imshow(
            after_display
        )

    ax2.set_title(
        "After"
    )

    ax2.axis("off")

    # --------------------------------------------------------

    ax3 = plt.subplot(2, 4, 3)

    ax3.imshow(
        target,
        cmap="gray"
    )

    ax3.set_title(
        "Ground Truth"
    )

    ax3.axis("off")

    # --------------------------------------------------------

    ax4 = plt.subplot(2, 4, 4)

    ax4.imshow(
        probability,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    ax4.set_title(
        f"Probability\nThreshold={THRESHOLD}"
    )

    ax4.axis("off")

    # --------------------------------------------------------

    ax5 = plt.subplot(2, 4, 5)

    ax5.imshow(
        prediction,
        cmap="gray"
    )

    ax5.set_title(
        "Prediction"
    )

    ax5.axis("off")

    # --------------------------------------------------------
    # FP
    # --------------------------------------------------------

    ax6 = plt.subplot(2, 4, 6)

    fp_display = np.zeros(
        (*target.shape, 3),
        dtype=np.float32
    )

    fp_display[fp] = [
        1.0,
        0.0,
        0.0
    ]

    ax6.imshow(
        fp_display
    )

    ax6.set_title(
        f"False Positives\nPixels={fp.sum():,}"
    )

    ax6.axis("off")

    # --------------------------------------------------------
    # FN
    # --------------------------------------------------------

    ax7 = plt.subplot(2, 4, 7)

    fn_display = np.zeros(
        (*target.shape, 3),
        dtype=np.float32
    )

    fn_display[fn] = [
        0.0,
        0.0,
        1.0
    ]

    ax7.imshow(
        fn_display
    )

    ax7.set_title(
        f"False Negatives\nPixels={fn.sum():,}"
    )

    ax7.axis("off")

    # --------------------------------------------------------
    # Combined
    # --------------------------------------------------------

    ax8 = plt.subplot(2, 4, 8)

    combined = np.zeros(
        (*target.shape, 3),
        dtype=np.float32
    )

    # TP
    combined[tp] = [
        0.0,
        1.0,
        0.0
    ]

    # FP
    combined[fp] = [
        1.0,
        0.0,
        0.0
    ]

    # FN
    combined[fn] = [
        0.0,
        0.0,
        1.0
    ]

    ax8.imshow(
        combined
    )

    ax8.set_title(
        "Error Map\n"
        "Green=TP Red=FP Blue=FN"
    )

    ax8.axis("off")

    # --------------------------------------------------------

    fig.suptitle(
        filename,
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# SAVE DISTRIBUTION PLOT
# ============================================================

def save_distribution_plot(

    values,
    title,
    xlabel,
    filename
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        values,
        bins=20
    )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        title
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = OUTPUT_DIR / filename

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# SAVE ERROR VS CHANGE AREA
# ============================================================

def save_error_vs_area_plot(

    areas,
    errors,
    ylabel,
    title,
    filename
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        areas,
        errors,
        alpha=0.7
    )

    plt.xlabel(
        "Ground Truth Change Area (%)"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = OUTPUT_DIR / filename

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(

    results,

    overall
):

    report_path = (
        OUTPUT_DIR /
        "error_type_analysis_report.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            "============================================================\n"
        )

        f.write(
            "URBAN CHANGE DETECTION\n"
        )

        f.write(
            "FALSE POSITIVE / FALSE NEGATIVE ERROR ANALYSIS\n"
        )

        f.write(
            "============================================================\n\n"
        )

        f.write(
            f"Threshold: {THRESHOLD}\n"
        )

        f.write(
            f"Images analyzed: {len(results)}\n\n"
        )

        f.write(
            "OVERALL ERROR STATISTICS\n"
        )

        f.write(
            "------------------------------------------------------------\n"
        )

        for key, value in overall.items():

            f.write(
                f"{key}: {value}\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Worst FP
        # ----------------------------------------------------

        sorted_fp = sorted(

            results,

            key=lambda x:
            x["fp_percentage"],

            reverse=True
        )

        f.write(
            "TOP 10 IMAGES BY FALSE POSITIVE PERCENTAGE\n"
        )

        f.write(
            "------------------------------------------------------------\n"
        )

        for i, item in enumerate(
            sorted_fp[:10],
            start=1
        ):

            f.write(

                f"{i}. "
                f"{item['filename']} | "
                f"FP={item['fp']:,} | "
                f"FP%={item['fp_percentage']:.4f}%\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Worst FN
        # ----------------------------------------------------

        sorted_fn = sorted(

            results,

            key=lambda x:
            x["fn_percentage"],

            reverse=True
        )

        f.write(
            "TOP 10 IMAGES BY FALSE NEGATIVE PERCENTAGE\n"
        )

        f.write(
            "------------------------------------------------------------\n"
        )

        for i, item in enumerate(
            sorted_fn[:10],
            start=1
        ):

            f.write(

                f"{i}. "
                f"{item['filename']} | "
                f"FN={item['fn']:,} | "
                f"FN%={item['fn_percentage']:.4f}%\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        f.write(
            "INTERPRETATION\n"
        )

        f.write(
            "------------------------------------------------------------\n"
        )

        f.write(
            "False Positives indicate areas where the model "
            "predicted change although the ground truth "
            "indicates no change.\n\n"
        )

        f.write(
            "False Negatives indicate actual changed pixels "
            "that were missed by the model.\n\n"
        )

        f.write(
            "Large FP values may indicate confusion caused "
            "by appearance changes, shadows, illumination, "
            "or similar structures.\n\n"
        )

        f.write(
            "Large FN values may indicate difficulty detecting "
            "small, subtle, fragmented, or low-contrast changes.\n"
        )

    print()
    print(
        f"Report saved:\n{report_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print()
    print("=" * 60)
    print("URBAN CHANGE DETECTION")
    print("FALSE POSITIVE / FALSE NEGATIVE ERROR ANALYSIS")
    print("=" * 60)

    device = get_device()

    print()
    print(f"Device: {device}")

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    check_required_files()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = create_test_dataset()

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = create_test_loader(
        dataset
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        device
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    load_checkpoint(
        model,
        device
    )

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RUNNING ERROR TYPE ANALYSIS")
    print("=" * 60)

    print()
    print(
        f"Threshold: {THRESHOLD}"
    )

    all_results = []

    # Store selected visualization data
    visualization_data = []

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader,
            start=1
        ):

            before = batch["before"].to(
                device
            )

            after = batch["after"].to(
                device
            )

            mask = batch["mask"].to(
                device
            )

            filenames = batch[
                "filename"
            ]

            logits = model(
                before,
                after
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            ).float()

            # ------------------------------------------------
            # Process each image
            # ------------------------------------------------

            batch_size = before.shape[0]

            for i in range(batch_size):

                prediction = (
                    predictions[i]
                )

                target = (
                    mask[i]
                )

                probability = (
                    probabilities[i]
                )

                metrics = calculate_metrics(

                    prediction
                    .detach()
                    .cpu()
                    .numpy(),

                    target
                    .detach()
                    .cpu()
                    .numpy()
                )

                filename = filenames[i]

                # ------------------------------------------------
                # Pixel counts
                # ------------------------------------------------

                total_pixels = (
                    prediction.numel()
                )

                ground_truth_change = int(
                    target.sum()
                    .item()
                )

                predicted_change = int(
                    prediction.sum()
                    .item()
                )

                fp = metrics["fp"]

                fn = metrics["fn"]

                tp = metrics["tp"]

                tn = metrics["tn"]

                # ------------------------------------------------
                # Percentages
                # ------------------------------------------------

                fp_percentage = safe_divide(

                    fp * 100,

                    total_pixels
                )

                fn_percentage = safe_divide(

                    fn * 100,

                    total_pixels
                )

                change_percentage = safe_divide(

                    ground_truth_change * 100,

                    total_pixels
                )

                predicted_change_percentage = safe_divide(

                    predicted_change * 100,

                    total_pixels
                )

                result = {

                    "filename": filename,

                    "tp": tp,

                    "tn": tn,

                    "fp": fp,

                    "fn": fn,

                    "precision": metrics[
                        "precision"
                    ],

                    "recall": metrics[
                        "recall"
                    ],

                    "f1": metrics[
                        "f1"
                    ],

                    "iou": metrics[
                        "iou"
                    ],

                    "accuracy": metrics[
                        "accuracy"
                    ],

                    "ground_truth_change_pixels":
                        ground_truth_change,

                    "predicted_change_pixels":
                        predicted_change,

                    "change_percentage":
                        change_percentage,

                    "predicted_change_percentage":
                        predicted_change_percentage,

                    "fp_percentage":
                        fp_percentage,

                    "fn_percentage":
                        fn_percentage
                }

                all_results.append(
                    result
                )

                # Keep tensors for later visualization
                visualization_data.append({

                    "filename": filename,

                    "before": before[i].detach().cpu(),

                    "after": after[i].detach().cpu(),

                    "target": target.detach().cpu(),

                    "prediction": prediction.detach().cpu(),

                    "probability": probability.detach().cpu(),

                    "metrics": result
                })

            print(
                f"Processed batch "
                f"{batch_index}/{len(loader)}"
            )

    # ========================================================
    # SAVE PER IMAGE RESULTS
    # ========================================================

    per_image_path = (
        OUTPUT_DIR /
        "per_image_error_types.json"
    )

    with open(
        per_image_path,
        "w"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4
        )

    print()
    print(
        f"Per-image results saved:\n"
        f"{per_image_path}"
    )

    # ========================================================
    # OVERALL STATISTICS
    # ========================================================

    total_tp = sum(
        x["tp"]
        for x in all_results
    )

    total_tn = sum(
        x["tn"]
        for x in all_results
    )

    total_fp = sum(
        x["fp"]
        for x in all_results
    )

    total_fn = sum(
        x["fn"]
        for x in all_results
    )

    total_pixels = (
        total_tp +
        total_tn +
        total_fp +
        total_fn
    )

    overall_precision = safe_divide(

        total_tp,

        total_tp + total_fp
    )

    overall_recall = safe_divide(

        total_tp,

        total_tp + total_fn
    )

    overall_f1 = safe_divide(

        2 *
        overall_precision *
        overall_recall,

        overall_precision +
        overall_recall
    )

    overall_iou = safe_divide(

        total_tp,

        total_tp +
        total_fp +
        total_fn
    )

    overall_accuracy = safe_divide(

        total_tp + total_tn,

        total_pixels
    )

    overall = {

        "total_pixels":
            total_pixels,

        "true_positive_pixels":
            total_tp,

        "true_negative_pixels":
            total_tn,

        "false_positive_pixels":
            total_fp,

        "false_negative_pixels":
            total_fn,

        "false_positive_percentage":
            safe_divide(
                total_fp * 100,
                total_pixels
            ),

        "false_negative_percentage":
            safe_divide(
                total_fn * 100,
                total_pixels
            ),

        "accuracy":
            overall_accuracy,

        "precision":
            overall_precision,

        "recall":
            overall_recall,

        "f1":
            overall_f1,

        "iou":
            overall_iou
    }

    # ========================================================
    # PRINT OVERALL RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("OVERALL ERROR STATISTICS")
    print("=" * 60)

    print()
    print(
        f"Total pixels : {total_pixels:,}"
    )

    print(
        f"TP           : {total_tp:,}"
    )

    print(
        f"TN           : {total_tn:,}"
    )

    print(
        f"FP           : {total_fp:,}"
    )

    print(
        f"FN           : {total_fn:,}"
    )

    print()

    print(
        f"FP percentage: "
        f"{overall['false_positive_percentage']:.4f}%"
    )

    print(
        f"FN percentage: "
        f"{overall['false_negative_percentage']:.4f}%"
    )

    print()

    print(
        f"Accuracy : {overall_accuracy:.6f}"
    )

    print(
        f"Precision: {overall_precision:.6f}"
    )

    print(
        f"Recall   : {overall_recall:.6f}"
    )

    print(
        f"F1       : {overall_f1:.6f}"
    )

    print(
        f"IoU      : {overall_iou:.6f}"
    )

    # ========================================================
    # SAVE STATISTICS
    # ========================================================

    statistics_path = (
        OUTPUT_DIR /
        "error_type_statistics.json"
    )

    with open(
        statistics_path,
        "w"
    ) as f:

        json.dump(
            overall,
            f,
            indent=4
        )

    print()
    print(
        f"Statistics saved:\n"
        f"{statistics_path}"
    )

    # ========================================================
    # FP DISTRIBUTION
    # ========================================================

    fp_values = [

        x["fp_percentage"]

        for x in all_results
    ]

    fn_values = [

        x["fn_percentage"]

        for x in all_results
    ]

    change_areas = [

        x["change_percentage"]

        for x in all_results
    ]

    save_distribution_plot(

        fp_values,

        "False Positive Distribution",

        "False Positive Pixels (%)",

        "fp_distribution.png"
    )

    save_distribution_plot(

        fn_values,

        "False Negative Distribution",

        "False Negative Pixels (%)",

        "fn_distribution.png"
    )

    # ========================================================
    # ERROR VS CHANGE AREA
    # ========================================================

    save_error_vs_area_plot(

        change_areas,

        fp_values,

        "False Positive Pixels (%)",

        "False Positive vs Ground Truth Change Area",

        "fp_vs_change_area.png"
    )

    save_error_vs_area_plot(

        change_areas,

        fn_values,

        "False Negative Pixels (%)",

        "False Negative vs Ground Truth Change Area",

        "fn_vs_change_area.png"
    )

    # ========================================================
    # FIND WORST FP
    # ========================================================

    worst_fp = sorted(

        visualization_data,

        key=lambda x:
        x["metrics"]["fp_percentage"],

        reverse=True
    )

    # ========================================================
    # FIND WORST FN
    # ========================================================

    worst_fn = sorted(

        visualization_data,

        key=lambda x:
        x["metrics"]["fn_percentage"],

        reverse=True
    )

    # ========================================================
    # PRINT WORST FP
    # ========================================================

    print()
    print("=" * 60)
    print("TOP 10 FALSE POSITIVE CASES")
    print("=" * 60)

    for index, item in enumerate(
        worst_fp[:10],
        start=1
    ):

        m = item["metrics"]

        print(

            f"{index}. "
            f"{item['filename']} | "
            f"FP={m['fp']:,} | "
            f"FP%={m['fp_percentage']:.4f}% | "
            f"F1={m['f1']:.4f}"
        )

    # ========================================================
    # PRINT WORST FN
    # ========================================================

    print()
    print("=" * 60)
    print("TOP 10 FALSE NEGATIVE CASES")
    print("=" * 60)

    for index, item in enumerate(
        worst_fn[:10],
        start=1
    ):

        m = item["metrics"]

        print(

            f"{index}. "
            f"{item['filename']} | "
            f"FN={m['fn']:,} | "
            f"FN%={m['fn_percentage']:.4f}% | "
            f"F1={m['f1']:.4f}"
        )

    # ========================================================
    # SAVE WORST FP VISUALIZATIONS
    # ========================================================

    print()
    print(
        "Generating highest FP visualizations..."
    )

    for item in worst_fp[:10]:

        filename = item["filename"]

        safe_name = Path(
            filename
        ).stem

        output_path = (
            WORST_FP_DIR /
            f"{safe_name}_FP_analysis.png"
        )

        save_error_visualization(

            item["before"],

            item["after"],

            item["target"],

            item["prediction"],

            item["probability"],

            filename,

            output_path
        )

        print(
            f"Saved: {output_path}"
        )

    # ========================================================
    # SAVE WORST FN VISUALIZATIONS
    # ========================================================

    print()
    print(
        "Generating highest FN visualizations..."
    )

    for item in worst_fn[:10]:

        filename = item["filename"]

        safe_name = Path(
            filename
        ).stem

        output_path = (
            WORST_FN_DIR /
            f"{safe_name}_FN_analysis.png"
        )

        save_error_visualization(

            item["before"],

            item["after"],

            item["target"],

            item["prediction"],

            item["probability"],

            filename,

            output_path
        )

        print(
            f"Saved: {output_path}"
        )

    # ========================================================
    # REPORT
    # ========================================================

    save_report(

        all_results,

        overall
    )

    # ========================================================
    # FINISH
    # ========================================================

    runtime = (
        time.time() -
        start_time
    )

    print()
    print("=" * 60)
    print("ERROR TYPE ANALYSIS COMPLETE 🎉")
    print("=" * 60)

    print()
    print(
        f"Threshold       : {THRESHOLD}"
    )

    print(
        f"Images analyzed : {len(all_results)}"
    )

    print(
        f"Runtime          : {runtime:.2f} seconds"
    )

    print()
    print(
        f"Results directory:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print(
        "Generated:"
    )

    print(
        f"Per-image results : "
        f"{per_image_path}"
    )

    print(
        f"Statistics        : "
        f"{statistics_path}"
    )

    print(
        f"Report            : "
        f"{OUTPUT_DIR / 'error_type_analysis_report.txt'}"
    )

    print(
        f"FP visualizations : "
        f"{WORST_FP_DIR}"
    )

    print(
        f"FN visualizations : "
        f"{WORST_FN_DIR}"
    )

    print(
        f"FP distribution   : "
        f"{OUTPUT_DIR / 'fp_distribution.png'}"
    )

    print(
        f"FN distribution   : "
        f"{OUTPUT_DIR / 'fn_distribution.png'}"
    )

    print(
        f"FP vs area        : "
        f"{OUTPUT_DIR / 'fp_vs_change_area.png'}"
    )

    print(
        f"FN vs area        : "
        f"{OUTPUT_DIR / 'fn_vs_change_area.png'}"
    )

    print()
    print(
        "Everything completed successfully. 🎉🔥"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()