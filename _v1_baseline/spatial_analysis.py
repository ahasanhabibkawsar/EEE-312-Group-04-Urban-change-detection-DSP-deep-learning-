import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset.levir_dataset import LEVIRCDDataset
from models.siamese_unet import SiameseUNet


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "data" / "LEVIR-CD"

TEST_A_DIR = DATASET_ROOT / "test" / "A"
TEST_B_DIR = DATASET_ROOT / "test" / "B"
TEST_LABEL_DIR = DATASET_ROOT / "test" / "label"

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "best_model.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "spatial_analysis"

IMAGE_SIZE = 256
BATCH_SIZE = 4
NUM_WORKERS = 0

THRESHOLD = 0.45


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

    print()
    print("All required files found. ✅")


# ============================================================
# CREATE DATASET
# ============================================================

def create_test_dataset():

    dataset = LEVIRCDDataset(
        image_a_dir=TEST_A_DIR,
        image_b_dir=TEST_B_DIR,
        label_dir=TEST_LABEL_DIR,
        image_size=IMAGE_SIZE,
        augment=False,
    )

    print()
    print(f"Valid test samples: {len(dataset)}")

    return dataset


# ============================================================
# CREATE DATALOADER
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
# CREATE MODEL
# ============================================================

def create_model(device):

    print()
    print("=" * 60)
    print("CREATING SIAMESE U-NET")
    print("=" * 60)

    model = SiameseUNet(
        in_channels=4,
    )

    model = model.to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model, device):

    print()
    print("=" * 60)
    print("LOADING CHECKPOINT")
    print("=" * 60)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
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

            model.load_state_dict(checkpoint)

        epoch = checkpoint.get(
            "epoch",
            "Unknown"
        )

        print(f"Checkpoint epoch: {epoch}")

    else:

        model.load_state_dict(checkpoint)

        print("Checkpoint format: state dictionary")

    print("Model loaded successfully. ✅")

    model.eval()

    print("Model switched to evaluation mode. ✅")

    return model


# ============================================================
# LOCATION CLASSIFICATION
# ============================================================

def classify_location(cx, cy):

    """
    Image coordinates:

        x -> horizontal
        y -> vertical

    3x3 spatial grid:

        TL | T  | TR
        ---+----+---
        L  | C  | R
        ---+----+---
        BL | B  | BR
    """

    if cx < 1 / 3 and cy < 1 / 3:

        return "Top-Left"

    elif cx >= 1 / 3 and cx < 2 / 3 and cy < 1 / 3:

        return "Top"

    elif cx >= 2 / 3 and cy < 1 / 3:

        return "Top-Right"

    elif cx < 1 / 3 and cy >= 1 / 3 and cy < 2 / 3:

        return "Left"

    elif cx >= 1 / 3 and cx < 2 / 3 and \
            cy >= 1 / 3 and cy < 2 / 3:

        return "Center"

    elif cx >= 2 / 3 and \
            cy >= 1 / 3 and cy < 2 / 3:

        return "Right"

    elif cx < 1 / 3 and cy >= 2 / 3:

        return "Bottom-Left"

    elif cx >= 1 / 3 and cx < 2 / 3 and cy >= 2 / 3:

        return "Bottom"

    else:

        return "Bottom-Right"


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, target):

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
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) /
        (tp + tn + fp + fn)
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
# CHANGE CENTROID
# ============================================================

def calculate_change_location(mask):

    """
    Returns:

        centroid_x
        centroid_y
        normalized_x
        normalized_y
        location

    If there is no change:
        location = "No change"
    """

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:

        return {
            "centroid_x": None,
            "centroid_y": None,
            "normalized_x": None,
            "normalized_y": None,
            "location": "No change",
        }

    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())

    normalized_x = centroid_x / mask.shape[1]
    normalized_y = centroid_y / mask.shape[0]

    location = classify_location(
        normalized_x,
        normalized_y
    )

    return {
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "normalized_x": normalized_x,
        "normalized_y": normalized_y,
        "location": location,
    }


# ============================================================
# ANALYZE DATASET
# ============================================================

@torch.no_grad()
def analyze_dataset(
    model,
    loader,
    device,
):

    all_results = []

    total_batches = len(loader)

    for batch_index, batch in enumerate(loader):

        before = batch["before"].to(device)
        after = batch["after"].to(device)
        masks = batch["mask"]

        filenames = batch["filename"]

        logits = model(
            before,
            after
        )

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= THRESHOLD
        ).float()

        predictions = predictions.cpu().numpy()
        masks = masks.cpu().numpy()

        probabilities = probabilities.cpu().numpy()

        batch_size = before.shape[0]

        for i in range(batch_size):

            prediction = predictions[i, 0]
            target = masks[i, 0]
            probability = probabilities[i, 0]

            filename = filenames[i]

            metrics = calculate_metrics(
                prediction,
                target
            )

            location_info = calculate_change_location(
                target
            )

            change_pixels = int(
                target.sum()
            )

            total_pixels = target.size

            change_percentage = (
                100.0 *
                change_pixels /
                total_pixels
            )

            result = {

                "filename": filename,

                "location": location_info["location"],

                "centroid_x":
                    location_info["centroid_x"],

                "centroid_y":
                    location_info["centroid_y"],

                "normalized_x":
                    location_info["normalized_x"],

                "normalized_y":
                    location_info["normalized_y"],

                "change_pixels":
                    change_pixels,

                "change_percentage":
                    float(change_percentage),

                "prediction_mean":
                    float(probability.mean()),

                "prediction_max":
                    float(probability.max()),

                "prediction_min":
                    float(probability.min()),

                **metrics,
            }

            all_results.append(result)

        print(
            f"Processed batch "
            f"{batch_index + 1}/{total_batches}"
        )

    return all_results


# ============================================================
# GROUP STATISTICS
# ============================================================

def calculate_location_statistics(results):

    categories = [
        "No change",
        "Top-Left",
        "Top",
        "Top-Right",
        "Left",
        "Center",
        "Right",
        "Bottom-Left",
        "Bottom",
        "Bottom-Right",
    ]

    statistics = {}

    for category in categories:

        samples = [
            r for r in results
            if r["location"] == category
        ]

        if len(samples) == 0:

            statistics[category] = {
                "samples": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "iou": 0.0,
                "change_percentage": 0.0,
            }

            continue

        statistics[category] = {

            "samples": len(samples),

            "precision": float(
                np.mean([
                    r["precision"]
                    for r in samples
                ])
            ),

            "recall": float(
                np.mean([
                    r["recall"]
                    for r in samples
                ])
            ),

            "f1": float(
                np.mean([
                    r["f1"]
                    for r in samples
                ])
            ),

            "iou": float(
                np.mean([
                    r["iou"]
                    for r in samples
                ])
            ),

            "change_percentage": float(
                np.mean([
                    r["change_percentage"]
                    for r in samples
                ])
            ),
        }

    return statistics


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    statistics,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    per_image_path = (
        OUTPUT_DIR /
        "per_image_spatial.json"
    )

    statistics_path = (
        OUTPUT_DIR /
        "spatial_statistics.json"
    )

    with open(
        per_image_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    with open(
        statistics_path,
        "w"
    ) as f:

        json.dump(
            statistics,
            f,
            indent=4
        )

    return (
        per_image_path,
        statistics_path,
    )


# ============================================================
# PLOT F1 BY LOCATION
# ============================================================

def plot_f1(statistics):

    categories = [
        "Top-Left",
        "Top",
        "Top-Right",
        "Left",
        "Center",
        "Right",
        "Bottom-Left",
        "Bottom",
        "Bottom-Right",
    ]

    values = [
        statistics[c]["f1"]
        for c in categories
    ]

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        categories,
        values
    )

    plt.xlabel(
        "Change Location"
    )

    plt.ylabel(
        "F1-score"
    )

    plt.title(
        "F1-score by Change Location"
    )

    plt.xticks(
        rotation=35
    )

    plt.ylim(
        0,
        1
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR /
        "f1_by_location.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(f"Saved: {path}")

    return path


# ============================================================
# PLOT IOU BY LOCATION
# ============================================================

def plot_iou(statistics):

    categories = [
        "Top-Left",
        "Top",
        "Top-Right",
        "Left",
        "Center",
        "Right",
        "Bottom-Left",
        "Bottom",
        "Bottom-Right",
    ]

    values = [
        statistics[c]["iou"]
        for c in categories
    ]

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        categories,
        values
    )

    plt.xlabel(
        "Change Location"
    )

    plt.ylabel(
        "IoU"
    )

    plt.title(
        "IoU by Change Location"
    )

    plt.xticks(
        rotation=35
    )

    plt.ylim(
        0,
        1
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR /
        "iou_by_location.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(f"Saved: {path}")

    return path


# ============================================================
# PLOT SAMPLE DISTRIBUTION
# ============================================================

def plot_distribution(statistics):

    categories = [
        "Top-Left",
        "Top",
        "Top-Right",
        "Left",
        "Center",
        "Right",
        "Bottom-Left",
        "Bottom",
        "Bottom-Right",
    ]

    values = [
        statistics[c]["samples"]
        for c in categories
    ]

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        categories,
        values
    )

    plt.xlabel(
        "Change Location"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.title(
        "Distribution of Change Locations"
    )

    plt.xticks(
        rotation=35
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR /
        "location_distribution.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(f"Saved: {path}")

    return path


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(
    statistics,
    results,
    runtime,
):

    report_path = (
        OUTPUT_DIR /
        "spatial_analysis_report.txt"
    )

    lines = []

    lines.append(
        "=" * 70
    )

    lines.append(
        "URBAN CHANGE DETECTION"
    )

    lines.append(
        "SPATIAL / LOCATION ERROR ANALYSIS"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    lines.append(
        f"Threshold: {THRESHOLD}"
    )

    lines.append(
        f"Images analyzed: {len(results)}"
    )

    lines.append(
        f"Runtime: {runtime:.2f} seconds"
    )

    lines.append("")

    lines.append(
        "LOCATION PERFORMANCE"
    )

    lines.append(
        "-" * 70
    )

    lines.append(
        f"{'Location':<15}"
        f"{'Samples':>10}"
        f"{'Change %':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>10}"
        f"{'IoU':>10}"
    )

    categories = [
        "No change",
        "Top-Left",
        "Top",
        "Top-Right",
        "Left",
        "Center",
        "Right",
        "Bottom-Left",
        "Bottom",
        "Bottom-Right",
    ]

    for category in categories:

        s = statistics[category]

        lines.append(
            f"{category:<15}"
            f"{s['samples']:>10}"
            f"{s['change_percentage']:>12.4f}"
            f"{s['precision']:>12.4f}"
            f"{s['recall']:>12.4f}"
            f"{s['f1']:>10.4f}"
            f"{s['iou']:>10.4f}"
        )

    lines.append("")

    # --------------------------------------------------------
    # BEST LOCATION
    # --------------------------------------------------------

    valid_categories = [
        c for c in categories
        if statistics[c]["samples"] > 0
        and c != "No change"
    ]

    if valid_categories:

        best_f1_category = max(
            valid_categories,
            key=lambda c:
                statistics[c]["f1"]
        )

        worst_f1_category = min(
            valid_categories,
            key=lambda c:
                statistics[c]["f1"]
        )

        best_iou_category = max(
            valid_categories,
            key=lambda c:
                statistics[c]["iou"]
        )

        worst_iou_category = min(
            valid_categories,
            key=lambda c:
                statistics[c]["iou"]
        )

        lines.append(
            "BEST / WORST LOCATIONS"
        )

        lines.append(
            "-" * 70
        )

        lines.append(
            f"Best F1 location  : "
            f"{best_f1_category} "
            f"({statistics[best_f1_category]['f1']:.4f})"
        )

        lines.append(
            f"Worst F1 location : "
            f"{worst_f1_category} "
            f"({statistics[worst_f1_category]['f1']:.4f})"
        )

        lines.append(
            f"Best IoU location : "
            f"{best_iou_category} "
            f"({statistics[best_iou_category]['iou']:.4f})"
        )

        lines.append(
            f"Worst IoU location: "
            f"{worst_iou_category} "
            f"({statistics[worst_iou_category]['iou']:.4f})"
        )

    lines.append("")

    lines.append(
        "=" * 70
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            "\n".join(lines)
        )

    print(
        f"Report saved: {report_path}"
    )

    return report_path


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print()
    print("=" * 70)
    print("URBAN CHANGE DETECTION")
    print("SPATIAL / LOCATION ERROR ANALYSIS")
    print("=" * 70)

    device = get_device()

    print()
    print(f"Device: {device}")

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    check_required_files()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CREATING TEST DATASET")
    print("=" * 60)

    dataset = create_test_dataset()

    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CREATING TEST DATALOADER")
    print("=" * 60)

    loader = create_dataloader(
        dataset
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model(
        device
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    model = load_checkpoint(
        model,
        device
    )

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RUNNING SPATIAL ANALYSIS")
    print("=" * 60)

    print()
    print(f"Threshold: {THRESHOLD}")

    results = analyze_dataset(
        model,
        loader,
        device,
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    print()
    print("Calculating spatial statistics...")

    statistics = calculate_location_statistics(
        results
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    per_image_path, statistics_path = save_results(
        results,
        statistics,
    )

    print()
    print(
        f"Per-image results saved:\n"
        f"{per_image_path}"
    )

    print(
        f"Statistics saved:\n"
        f"{statistics_path}"
    )

    # --------------------------------------------------------
    # PRINT TABLE
    # --------------------------------------------------------

    print()
    print("=" * 100)

    print(
        f"{'Category':<15}"
        f"{'Samples':>10}"
        f"{'Change %':>12}"
        f"{'Precision':>14}"
        f"{'Recall':>12}"
        f"{'F1':>10}"
        f"{'IoU':>10}"
    )

    print("=" * 100)

    categories = [
        "No change",
        "Top-Left",
        "Top",
        "Top-Right",
        "Left",
        "Center",
        "Right",
        "Bottom-Left",
        "Bottom",
        "Bottom-Right",
    ]

    for category in categories:

        s = statistics[category]

        print(
            f"{category:<15}"
            f"{s['samples']:>10}"
            f"{s['change_percentage']:>12.4f}"
            f"{s['precision']:>14.4f}"
            f"{s['recall']:>12.4f}"
            f"{s['f1']:>10.4f}"
            f"{s['iou']:>10.4f}"
        )

    print("=" * 100)

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    print()
    print("Generating plots...")

    plot_f1(
        statistics
    )

    plot_iou(
        statistics
    )

    plot_distribution(
        statistics
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    runtime = time.time() - start_time

    generate_report(
        statistics,
        results,
        runtime,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "SPATIAL ANALYSIS COMPLETE. 🎉"
    )

    print()

    print(
        f"Threshold: {THRESHOLD}"
    )

    print(
        f"Images analyzed: {len(results)}"
    )

    print(
        f"Runtime: {runtime:.2f} seconds"
    )

    print()

    print(
        f"Results directory:\n"
        f"{OUTPUT_DIR}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()