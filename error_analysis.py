import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# ------------------------------------------------------------
# PROJECT IMPORTS
# ------------------------------------------------------------

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

OUTPUT_DIR = PROJECT_ROOT / "error_analysis"

BEST_DIR = OUTPUT_DIR / "best_predictions"
WORST_DIR = OUTPUT_DIR / "worst_predictions"

BATCH_SIZE = 4

NUM_WORKERS = 0

IMAGE_SIZE = 256

THRESHOLD = load_best_threshold()  # v2: threshold selected on the VALIDATION set

TOP_K = 10


# ============================================================
# DEVICE
# ============================================================

def get_device():
    """
    Select the best available device.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_output_directories():
    """
    Create directories required for error analysis.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    BEST_DIR.mkdir(parents=True, exist_ok=True)

    WORST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILE CHECK
# ============================================================

def check_required_files():

    print("=" * 60)
    print("CHECKING REQUIRED FILES")
    print("=" * 60)

    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Test A       : {TEST_A_DIR}")
    print(f"Test B       : {TEST_B_DIR}")
    print(f"Test labels  : {TEST_LABEL_DIR}")
    print(f"Checkpoint   : {CHECKPOINT_PATH}")

    required_paths = [
        DATASET_ROOT,
        TEST_A_DIR,
        TEST_B_DIR,
        TEST_LABEL_DIR,
        CHECKPOINT_PATH,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired path not found:\n{path}"
            )

    print("\nAll required files found. ✅")


# ============================================================
# DATASET
# ============================================================

def create_test_dataset():

    print("\n" + "=" * 60)
    print("CREATING TEST DATASET")
    print("=" * 60)

    dataset = LEVIRCDDataset(
        image_a_dir=TEST_A_DIR,
        image_b_dir=TEST_B_DIR,
        label_dir=TEST_LABEL_DIR,
        image_size=IMAGE_SIZE,
        augment=False,
    )

    print(f"Valid test samples: {len(dataset)}")

    return dataset


# ============================================================
# DATALOADER
# ============================================================

def create_dataloader(dataset):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    print(f"Test batches: {len(loader)}")

    return loader


# ============================================================
# MODEL
# ============================================================

def create_model(device):

    print("\n" + "=" * 60)
    print("CREATING SIAMESE U-NET")
    print("=" * 60)

    # IMPORTANT:
    # This follows the constructor that has already
    # worked in your project.

    model = SiameseUNet(
        in_channels=4,
        base_channels=32,
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

    print(f"Total parameters    : {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model(model, device):

    print("\n" + "=" * 60)
    print("LOADING TRAINED MODEL")
    print("=" * 60)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    # --------------------------------------------------------
    # Different checkpoint formats are supported.
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            # Sometimes checkpoint itself is state_dict
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove possible DataParallel prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            new_key = key[len("module."):]

        else:

            new_key = key

        cleaned_state_dict[new_key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.eval()

    checkpoint_epoch = None

    if isinstance(checkpoint, dict):

        checkpoint_epoch = checkpoint.get(
            "epoch",
            None,
        )

    print(f"Checkpoint epoch: {checkpoint_epoch}")

    print("Model loaded successfully. ✅")

    print("Model switched to evaluation mode. ✅")

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, target):

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(
        prediction,
        target,
    ).sum()

    tn = np.logical_and(
        ~prediction,
        ~target,
    ).sum()

    fp = np.logical_and(
        prediction,
        ~target,
    ).sum()

    fn = np.logical_and(
        ~prediction,
        target,
    ).sum()

    epsilon = 1e-8

    accuracy = (
        (tp + tn)
        /
        (tp + tn + fp + fn + epsilon)
    )

    precision = (
        tp
        /
        (tp + fp + epsilon)
    )

    recall = (
        tp
        /
        (tp + fn + epsilon)
    )

    f1 = (
        2 * precision * recall
        /
        (precision + recall + epsilon)
    )

    iou = (
        tp
        /
        (tp + fp + fn + epsilon)
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

def prepare_image_for_display(image):
    """
    Standardised model-input tensor -> RGB image in [0, 1]
    (v2: exact de-normalisation instead of min-max scaling).
    """

    return denormalize_for_display(image)


def create_error_map(prediction, target):

    prediction = prediction.astype(bool)

    target = target.astype(bool)

    error_map = np.zeros(
        prediction.shape,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # 0 = TN
    # 1 = TP
    # 2 = FP
    # 3 = FN
    # --------------------------------------------------------

    tp = prediction & target

    fp = prediction & (~target)

    fn = (~prediction) & target

    error_map[tp] = 1

    error_map[fp] = 2

    error_map[fn] = 3

    return error_map


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(
    before,
    after,
    target,
    prediction,
    probability,
    filename,
    metrics,
    output_directory,
):

    before_display = prepare_image_for_display(
        before
    )

    after_display = prepare_image_for_display(
        after
    )

    target = target.astype(np.uint8)

    prediction = prediction.astype(np.uint8)

    error_map = create_error_map(
        prediction,
        target,
    )

    probability = probability.astype(
        np.float32
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
    )

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    axes[0, 0].imshow(before_display)

    axes[0, 0].set_title(
        "Before Image"
    )

    axes[0, 0].axis("off")

    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------

    axes[0, 1].imshow(after_display)

    axes[0, 1].set_title(
        "After Image"
    )

    axes[0, 1].axis("off")

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    axes[0, 2].imshow(
        target,
        cmap="gray",
    )

    axes[0, 2].set_title(
        "Ground Truth"
    )

    axes[0, 2].axis("off")

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    axes[0, 3].imshow(
        prediction,
        cmap="gray",
    )

    axes[0, 3].set_title(
        f"Prediction @ {THRESHOLD}"
    )

    axes[0, 3].axis("off")

    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    axes[1, 0].imshow(
        probability,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    axes[1, 0].set_title(
        "Change Probability"
    )

    axes[1, 0].axis("off")

    # --------------------------------------------------------
    # ERROR MAP
    # --------------------------------------------------------

    axes[1, 1].imshow(
        error_map,
        cmap="viridis",
        vmin=0,
        vmax=3,
    )

    axes[1, 1].set_title(
        "Error Map"
    )

    axes[1, 1].axis("off")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    axes[1, 2].axis("off")

    metric_text = (
        f"F1      : {metrics['f1']:.4f}\n"
        f"IoU     : {metrics['iou']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall  : {metrics['recall']:.4f}\n\n"
        f"TP: {metrics['tp']}\n"
        f"TN: {metrics['tn']}\n"
        f"FP: {metrics['fp']}\n"
        f"FN: {metrics['fn']}"
    )

    axes[1, 2].text(
        0.05,
        0.5,
        metric_text,
        fontsize=11,
        verticalalignment="center",
    )

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    axes[1, 3].axis("off")

    legend_text = (
        "ERROR MAP\n\n"
        "0 = True Negative\n"
        "1 = True Positive\n"
        "2 = False Positive\n"
        "3 = False Negative"
    )

    axes[1, 3].text(
        0.05,
        0.5,
        legend_text,
        fontsize=11,
        verticalalignment="center",
    )

    fig.suptitle(
        filename,
        fontsize=14,
    )

    plt.tight_layout()

    output_path = (
        output_directory
        /
        f"{Path(filename).stem}_error_analysis.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


# ============================================================
# SAVE JSON
# ============================================================

def save_json(data, path):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


# ============================================================
# RUN INFERENCE
# ============================================================

def run_error_analysis(
    model,
    loader,
    device,
):

    print("\n" + "=" * 60)
    print("RUNNING ERROR ANALYSIS")
    print("=" * 60)

    results = []

    total_batches = len(loader)

    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader,
            start=1,
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

            filenames = batch["filename"]

            logits = model(
                before,
                after,
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            ).float()

            for i in range(
                before.shape[0]
            ):

                filename = filenames[i]

                prediction_np = (
                    predictions[i, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                target_np = (
                    mask[i, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                probability_np = (
                    probabilities[i, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                metrics = calculate_metrics(
                    prediction_np,
                    target_np,
                )

                result = {
                    "filename": filename,
                    **metrics,
                }

                results.append(result)

            print(
                f"Processed batch "
                f"{batch_index}/{total_batches}"
            )

    return results


# ============================================================
# SORT RESULTS
# ============================================================

def get_best_and_worst(results):

    best = sorted(
        results,
        key=lambda x: x["f1"],
        reverse=True,
    )

    worst = sorted(
        results,
        key=lambda x: x["f1"],
    )

    return (
        best[:TOP_K],
        worst[:TOP_K],
    )


# ============================================================
# SAVE RESULTS TABLE
# ============================================================

def save_results_table(results):

    path = (
        OUTPUT_DIR
        /
        "per_image_metrics.json"
    )

    save_json(
        results,
        path,
    )

    return path


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(results):

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
                item[metric]
                for item in results
            ],
            dtype=np.float64,
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
# CREATE ERROR DISTRIBUTION PLOT
# ============================================================

def create_error_distribution(
    results
):

    f1_values = [
        item["f1"]
        for item in results
    ]

    iou_values = [
        item["iou"]
        for item in results
    ]

    fig = plt.figure(
        figsize=(12, 5)
    )

    plt.hist(
        f1_values,
        bins=20,
    )

    plt.xlabel("F1-score")

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Per-image F1 Distribution"
    )

    plt.grid(
        alpha=0.3
    )

    path = (
        OUTPUT_DIR
        /
        "f1_distribution.png"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # IoU distribution
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(12, 5)
    )

    plt.hist(
        iou_values,
        bins=20,
    )

    plt.xlabel("IoU")

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Per-image IoU Distribution"
    )

    plt.grid(
        alpha=0.3
    )

    path_iou = (
        OUTPUT_DIR
        /
        "iou_distribution.png"
    )

    plt.tight_layout()

    plt.savefig(
        path_iou,
        dpi=150,
    )

    plt.close(fig)

    return path, path_iou


# ============================================================
# SAVE TEXT REPORT
# ============================================================

def save_text_report(
    results,
    statistics,
    best,
    worst,
):

    path = (
        OUTPUT_DIR
        /
        "error_analysis_report.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "=" * 60
            + "\n"
        )

        file.write(
            "URBAN CHANGE DETECTION\n"
        )

        file.write(
            "ERROR ANALYSIS REPORT\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            f"Number of test images: "
            f"{len(results)}\n"
        )

        file.write(
            f"Threshold: "
            f"{THRESHOLD}\n\n"
        )

        file.write(
            "OVERALL PER-IMAGE STATISTICS\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        for metric, values in statistics.items():

            file.write(
                f"{metric.upper():10s} "
                f"Mean={values['mean']:.4f}  "
                f"Std={values['std']:.4f}  "
                f"Min={values['minimum']:.4f}  "
                f"Max={values['maximum']:.4f}\n"
            )

        file.write("\n")

        file.write(
            "TOP 10 BEST IMAGES\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        for index, item in enumerate(
            best,
            start=1,
        ):

            file.write(
                f"{index}. "
                f"{item['filename']} | "
                f"F1={item['f1']:.4f} | "
                f"IoU={item['iou']:.4f} | "
                f"Precision={item['precision']:.4f} | "
                f"Recall={item['recall']:.4f}\n"
            )

        file.write("\n")

        file.write(
            "TOP 10 WORST IMAGES\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        for index, item in enumerate(
            worst,
            start=1,
        ):

            file.write(
                f"{index}. "
                f"{item['filename']} | "
                f"F1={item['f1']:.4f} | "
                f"IoU={item['iou']:.4f} | "
                f"Precision={item['precision']:.4f} | "
                f"Recall={item['recall']:.4f}\n"
            )

        file.write("\n")

        zero_f1 = [
            item
            for item in results
            if item["f1"] == 0
        ]

        file.write(
            "ZERO-F1 IMAGES\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        file.write(
            f"Count: {len(zero_f1)}\n\n"
        )

        for item in zero_f1:

            file.write(
                f"{item['filename']}\n"
            )

            file.write(
                f"  TP={item['tp']} "
                f"TN={item['tn']} "
                f"FP={item['fp']} "
                f"FN={item['fn']}\n"
            )

    return path


# ============================================================
# GENERATE SELECTED VISUALIZATIONS
# ============================================================

def generate_selected_visualizations(
    model,
    dataset,
    selected_results,
    output_directory,
    device,
):

    print(
        "\nGenerating selected visualizations..."
    )

    filename_to_index = {
        filename: index
        for index, filename in enumerate(
            dataset.file_names
        )
    }

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    selected_names = {
        item["filename"]
        for item in selected_results
    }

    with torch.no_grad():

        for batch in loader:

            filename = batch["filename"][0]

            if filename not in selected_names:

                continue

            before = batch["before"].to(
                device
            )

            after = batch["after"].to(
                device
            )

            mask = batch["mask"].to(
                device
            )

            logits = model(
                before,
                after,
            )

            probability = torch.sigmoid(
                logits
            )

            prediction = (
                probability >= THRESHOLD
            )

            result = next(
                item
                for item in selected_results
                if item["filename"] == filename
            )

            output_path = save_visualization(
                before[0],
                after[0],
                mask[0, 0].cpu().numpy(),
                prediction[0, 0].cpu().numpy(),
                probability[0, 0].cpu().numpy(),
                filename,
                result,
                output_directory,
            )

            print(
                f"Saved: {output_path}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 60)
    print("URBAN CHANGE DETECTION")
    print("ERROR ANALYSIS")
    print("=" * 60)

    device = get_device()

    print(f"\nDevice: {device}")

    create_output_directories()

    check_required_files()

    dataset = create_test_dataset()

    loader = create_dataloader(
        dataset
    )

    model = create_model(
        device
    )

    model = load_model(
        model,
        device,
    )

    # --------------------------------------------------------
    # RUN ANALYSIS
    # --------------------------------------------------------

    results = run_error_analysis(
        model,
        loader,
        device,
    )

    print(
        f"\nImages analyzed: "
        f"{len(results)}"
    )

    # --------------------------------------------------------
    # SAVE PER IMAGE RESULTS
    # --------------------------------------------------------

    json_path = save_results_table(
        results
    )

    print(
        f"Per-image metrics saved:\n"
        f"{json_path}"
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    statistics = calculate_statistics(
        results
    )

    statistics_path = (
        OUTPUT_DIR
        /
        "statistics.json"
    )

    save_json(
        statistics,
        statistics_path,
    )

    # --------------------------------------------------------
    # BEST / WORST
    # --------------------------------------------------------

    best, worst = get_best_and_worst(
        results
    )

    print("\n" + "=" * 60)
    print("BEST 10 SAMPLES")
    print("=" * 60)

    for index, item in enumerate(
        best,
        start=1,
    ):

        print(
            f"{index}. "
            f"{item['filename']} | "
            f"F1={item['f1']:.4f} | "
            f"IoU={item['iou']:.4f}"
        )

    print("\n" + "=" * 60)
    print("WORST 10 SAMPLES")
    print("=" * 60)

    for index, item in enumerate(
        worst,
        start=1,
    ):

        print(
            f"{index}. "
            f"{item['filename']} | "
            f"F1={item['f1']:.4f} | "
            f"IoU={item['iou']:.4f}"
        )

    # --------------------------------------------------------
    # ZERO F1
    # --------------------------------------------------------

    zero_f1 = [
        item
        for item in results
        if item["f1"] == 0
    ]

    print("\n" + "=" * 60)
    print("ZERO-F1 SAMPLES")
    print("=" * 60)

    if len(zero_f1) == 0:

        print("No zero-F1 samples found. ✅")

    else:

        for item in zero_f1:

            print(
                f"{item['filename']} | "
                f"TP={item['tp']} "
                f"TN={item['tn']} "
                f"FP={item['fp']} "
                f"FN={item['fn']}"
            )

    # --------------------------------------------------------
    # DISTRIBUTION PLOTS
    # --------------------------------------------------------

    f1_plot, iou_plot = (
        create_error_distribution(
            results
        )
    )

    print(
        f"\nSaved: {f1_plot}"
    )

    print(
        f"Saved: {iou_plot}"
    )

    # --------------------------------------------------------
    # TEXT REPORT
    # --------------------------------------------------------

    report_path = save_text_report(
        results,
        statistics,
        best,
        worst,
    )

    print(
        f"\nReport saved:\n"
        f"{report_path}"
    )

    # --------------------------------------------------------
    # BEST VISUALIZATIONS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "GENERATING BEST SAMPLE VISUALIZATIONS"
    )

    print(
        "=" * 60
    )

    generate_selected_visualizations(
        model,
        dataset,
        best,
        BEST_DIR,
        device,
    )

    # --------------------------------------------------------
    # WORST VISUALIZATIONS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "GENERATING WORST SAMPLE VISUALIZATIONS"
    )

    print(
        "=" * 60
    )

    generate_selected_visualizations(
        model,
        dataset,
        worst,
        WORST_DIR,
        device,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    elapsed = (
        time.time()
        -
        start_time
    )

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS COMPLETE ✅")
    print("=" * 60)

    print(
        f"\nThreshold used: {THRESHOLD}"
    )

    print(
        f"Images analyzed: {len(results)}"
    )

    print(
        f"Runtime: {elapsed:.2f} seconds"
    )

    print(
        f"\nResults directory:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        "\nGenerated:"
    )

    print(
        f"  Per-image metrics : "
        f"{json_path}"
    )

    print(
        f"  Statistics        : "
        f"{statistics_path}"
    )

    print(
        f"  Report            : "
        f"{report_path}"
    )

    print(
        f"  Best samples      : "
        f"{BEST_DIR}"
    )

    print(
        f"  Worst samples     : "
        f"{WORST_DIR}"
    )

    print(
        f"  F1 distribution   : "
        f"{f1_plot}"
    )

    print(
        f"  IoU distribution  : "
        f"{iou_plot}"
    )

    print(
        "\nEverything completed successfully. 🎉"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()