import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

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

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

OUTPUT_DIR = PROJECT_ROOT / "change_size_analysis"

IMAGE_SIZE = 256
BATCH_SIZE = 4

# Best threshold obtained from previous threshold analysis
THRESHOLD = 0.45

NUM_WORKERS = 0


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
# PRINT HEADER
# ============================================================

def print_header(title):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# CHECK FILES
# ============================================================

def check_required_files():

    print_header("CHECKING REQUIRED FILES")

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
# CREATE DATASET
# ============================================================

def create_test_dataset():

    print_header("CREATING TEST DATASET")

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
# CREATE DATALOADER
# ============================================================

def create_dataloader(dataset):

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(device):

    print_header("CREATING SIAMESE U-NET")

    model = SiameseUNet(
        in_channels=4,
    )

    model = model.to(device)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters    : {total_parameters:,}"
    )

    print(
        f"Trainable parameters: {trainable_parameters:,}"
    )

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model(model, device):

    print_header("LOADING TRAINED MODEL")

    print(f"Checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    # --------------------------------------------------------
    # Handle different checkpoint formats
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

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    if isinstance(checkpoint, dict):

        epoch = checkpoint.get(
            "epoch",
            "Unknown"
        )

    else:

        epoch = "Unknown"

    print(f"Checkpoint epoch: {epoch}")

    print("Model loaded successfully. ✅")

    model.eval()

    print("Model switched to evaluation mode. ✅")

    return model


# ============================================================
# CALCULATE METRICS
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

    if precision + recall > 0:

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn)
        / (tp + tn + fp + fn)
    )

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


# ============================================================
# CHANGE SIZE CATEGORY
# ============================================================

def get_change_category(change_percentage):

    # Percentage of image occupied by ground-truth change

    if change_percentage == 0:

        return "No change"

    elif change_percentage < 0.5:

        return "Very small"

    elif change_percentage < 1.0:

        return "Small"

    elif change_percentage < 2.0:

        return "Medium"

    else:

        return "Large"


# ============================================================
# ANALYZE DATASET
# ============================================================

def analyze_dataset(
    model,
    dataloader,
    device,
):

    print_header("RUNNING CHANGE-SIZE ERROR ANALYSIS")

    print(f"Threshold: {THRESHOLD}")

    all_results = []

    model.eval()

    total_batches = len(dataloader)

    start_time = time.time()

    with torch.no_grad():

        for batch_index, batch in enumerate(
            dataloader,
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

            filenames = batch["filename"]

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

            batch_size = before.shape[0]

            for i in range(batch_size):

                prediction = (
                    predictions[i]
                    .detach()
                    .cpu()
                    .numpy()
                )

                target = (
                    mask[i]
                    .detach()
                    .cpu()
                    .numpy()
                )

                prediction = np.squeeze(
                    prediction
                )

                target = np.squeeze(
                    target
                )

                # ------------------------------------------------
                # Ground truth change area
                # ------------------------------------------------

                total_pixels = target.size

                changed_pixels = np.sum(
                    target > 0.5
                )

                change_percentage = (
                    changed_pixels
                    / total_pixels
                    * 100.0
                )

                category = get_change_category(
                    change_percentage
                )

                # ------------------------------------------------
                # Metrics
                # ------------------------------------------------

                metrics = calculate_metrics(
                    prediction > 0.5,
                    target > 0.5,
                )

                result = {

                    "filename": str(
                        filenames[i]
                    ),

                    "total_pixels": int(
                        total_pixels
                    ),

                    "changed_pixels": int(
                        changed_pixels
                    ),

                    "change_percentage":
                        float(
                            change_percentage
                        ),

                    "category": category,

                    **metrics,
                }

                all_results.append(result)

            print(
                f"Processed batch "
                f"{batch_index}/{total_batches}"
            )

    runtime = time.time() - start_time

    print(
        f"\nImages analyzed: "
        f"{len(all_results)}"
    )

    print(
        f"Runtime: {runtime:.2f} seconds"
    )

    return all_results


# ============================================================
# CALCULATE GROUP STATISTICS
# ============================================================

def calculate_group_statistics(results):

    print_header("CALCULATING CHANGE-SIZE STATISTICS")

    categories = [
        "No change",
        "Very small",
        "Small",
        "Medium",
        "Large",
    ]

    statistics = {}

    for category in categories:

        samples = [
            r
            for r in results
            if r["category"] == category
        ]

        count = len(samples)

        if count == 0:

            statistics[category] = {
                "count": 0,
                "mean_change_percentage": 0.0,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
                "mean_f1": 0.0,
                "mean_iou": 0.0,
                "mean_accuracy": 0.0,
            }

            continue

        statistics[category] = {

            "count": count,

            "mean_change_percentage":
                float(
                    np.mean([
                        r["change_percentage"]
                        for r in samples
                    ])
                ),

            "mean_precision":
                float(
                    np.mean([
                        r["precision"]
                        for r in samples
                    ])
                ),

            "mean_recall":
                float(
                    np.mean([
                        r["recall"]
                        for r in samples
                    ])
                ),

            "mean_f1":
                float(
                    np.mean([
                        r["f1"]
                        for r in samples
                    ])
                ),

            "mean_iou":
                float(
                    np.mean([
                        r["iou"]
                        for r in samples
                    ])
                ),

            "mean_accuracy":
                float(
                    np.mean([
                        r["accuracy"]
                        for r in samples
                    ])
                ),
        }

    return statistics


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_statistics(statistics):

    print_header("CHANGE-SIZE PERFORMANCE")

    print(
        f"{'Category':<15}"
        f"{'Samples':>8}"
        f"{'Change %':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'IoU':>12}"
    )

    print("-" * 85)

    order = [
        "No change",
        "Very small",
        "Small",
        "Medium",
        "Large",
    ]

    for category in order:

        s = statistics[category]

        print(
            f"{category:<15}"
            f"{s['count']:>8}"
            f"{s['mean_change_percentage']:>12.4f}"
            f"{s['mean_precision']:>12.4f}"
            f"{s['mean_recall']:>12.4f}"
            f"{s['mean_f1']:>12.4f}"
            f"{s['mean_iou']:>12.4f}"
        )


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
        OUTPUT_DIR
        / "per_image_change_size.json"
    )

    statistics_path = (
        OUTPUT_DIR
        / "change_size_statistics.json"
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

    print(
        f"\nPer-image results saved:"
        f"\n{per_image_path}"
    )

    print(
        f"\nStatistics saved:"
        f"\n{statistics_path}"
    )


# ============================================================
# PLOT F1 VS CHANGE SIZE
# ============================================================

def plot_f1_by_category(statistics):

    categories = [
        "No change",
        "Very small",
        "Small",
        "Medium",
        "Large",
    ]

    f1_values = [
        statistics[c]["mean_f1"]
        for c in categories
    ]

    output_path = (
        OUTPUT_DIR
        / "f1_by_change_size.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        categories,
        f1_values
    )

    plt.xlabel(
        "Ground-truth Change Size"
    )

    plt.ylabel(
        "Mean F1-score"
    )

    plt.title(
        "F1-score by Ground-truth Change Size"
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# PLOT IOU VS CHANGE SIZE
# ============================================================

def plot_iou_by_category(statistics):

    categories = [
        "No change",
        "Very small",
        "Small",
        "Medium",
        "Large",
    ]

    iou_values = [
        statistics[c]["mean_iou"]
        for c in categories
    ]

    output_path = (
        OUTPUT_DIR
        / "iou_by_change_size.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        categories,
        iou_values
    )

    plt.xlabel(
        "Ground-truth Change Size"
    )

    plt.ylabel(
        "Mean IoU"
    )

    plt.title(
        "IoU by Ground-truth Change Size"
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# PLOT CHANGE AREA VS F1
# ============================================================

def plot_change_area_vs_f1(results):

    x = [
        r["change_percentage"]
        for r in results
    ]

    y = [
        r["f1"]
        for r in results
    ]

    output_path = (
        OUTPUT_DIR
        / "change_area_vs_f1.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        x,
        y,
        alpha=0.7
    )

    plt.xlabel(
        "Ground-truth Change Area (%)"
    )

    plt.ylabel(
        "F1-score"
    )

    plt.title(
        "F1-score vs Ground-truth Change Area"
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# PLOT CHANGE AREA VS IOU
# ============================================================

def plot_change_area_vs_iou(results):

    x = [
        r["change_percentage"]
        for r in results
    ]

    y = [
        r["iou"]
        for r in results
    ]

    output_path = (
        OUTPUT_DIR
        / "change_area_vs_iou.png"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        x,
        y,
        alpha=0.7
    )

    plt.xlabel(
        "Ground-truth Change Area (%)"
    )

    plt.ylabel(
        "IoU"
    )

    plt.title(
        "IoU vs Ground-truth Change Area"
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(
    results,
    statistics,
):

    report_path = (
        OUTPUT_DIR
        / "change_size_analysis_report.txt"
    )

    categories = [
        "No change",
        "Very small",
        "Small",
        "Medium",
        "Large",
    ]

    with open(
        report_path,
        "w"
    ) as f:

        f.write(
            "URBAN CHANGE DETECTION\n"
        )

        f.write(
            "CHANGE-SIZE ERROR ANALYSIS\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"Total images analyzed: "
            f"{len(results)}\n"
        )

        f.write(
            f"Threshold: {THRESHOLD}\n\n"
        )

        f.write(
            "CHANGE-SIZE PERFORMANCE\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        for category in categories:

            s = statistics[category]

            f.write(
                f"\n{category}\n"
            )

            f.write(
                f"Samples: {s['count']}\n"
            )

            f.write(
                "Mean change percentage: "
                f"{s['mean_change_percentage']:.4f}%\n"
            )

            f.write(
                f"Precision: "
                f"{s['mean_precision']:.4f}\n"
            )

            f.write(
                f"Recall: "
                f"{s['mean_recall']:.4f}\n"
            )

            f.write(
                f"F1: "
                f"{s['mean_f1']:.4f}\n"
            )

            f.write(
                f"IoU: "
                f"{s['mean_iou']:.4f}\n"
            )

            f.write(
                f"Accuracy: "
                f"{s['mean_accuracy']:.4f}\n"
            )

        # ----------------------------------------------------
        # Best and worst samples
        # ----------------------------------------------------

        sorted_results = sorted(
            results,
            key=lambda x: x["f1"],
            reverse=True
        )

        f.write(
            "\n\nTOP 10 BEST SAMPLES\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        for index, r in enumerate(
            sorted_results[:10],
            start=1
        ):

            f.write(
                f"{index}. "
                f"{r['filename']} | "
                f"Change={r['change_percentage']:.4f}% | "
                f"F1={r['f1']:.4f} | "
                f"IoU={r['iou']:.4f}\n"
            )

        f.write(
            "\n\nTOP 10 WORST SAMPLES\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        for index, r in enumerate(
            sorted_results[-10:],
            start=1
        ):

            f.write(
                f"{index}. "
                f"{r['filename']} | "
                f"Change={r['change_percentage']:.4f}% | "
                f"F1={r['f1']:.4f} | "
                f"IoU={r['iou']:.4f}\n"
            )

    print(
        f"\nReport saved:\n{report_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "URBAN CHANGE DETECTION"
    )

    print(
        "CHANGE-SIZE ERROR ANALYSIS"
    )

    device = get_device()

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Prepare directories
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    check_required_files()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = create_test_dataset()

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataloader = create_dataloader(
        dataset
    )

    print(
        f"Test batches: "
        f"{len(dataloader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        device
    )

    model = load_model(
        model,
        device
    )

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    results = analyze_dataset(
        model,
        dataloader,
        device,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    statistics = calculate_group_statistics(
        results
    )

    print_statistics(
        statistics
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    save_results(
        results,
        statistics
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    print_header(
        "GENERATING PLOTS"
    )

    plot_f1_by_category(
        statistics
    )

    plot_iou_by_category(
        statistics
    )

    plot_change_area_vs_f1(
        results
    )

    plot_change_area_vs_iou(
        results
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    generate_report(
        results,
        statistics
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print_header(
        "CHANGE-SIZE ANALYSIS COMPLETE"
    )

    print(
        f"Results directory:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        "\nGenerated:"
    )

    print(
        f"  Per-image results:\n"
        f"  {OUTPUT_DIR / 'per_image_change_size.json'}"
    )

    print(
        f"\n  Statistics:\n"
        f"  {OUTPUT_DIR / 'change_size_statistics.json'}"
    )

    print(
        f"\n  Report:\n"
        f"  {OUTPUT_DIR / 'change_size_analysis_report.txt'}"
    )

    print(
        f"\n  F1 plot:\n"
        f"  {OUTPUT_DIR / 'f1_by_change_size.png'}"
    )

    print(
        f"\n  IoU plot:\n"
        f"  {OUTPUT_DIR / 'iou_by_change_size.png'}"
    )

    print(
        f"\n  F1 vs area:\n"
        f"  {OUTPUT_DIR / 'change_area_vs_f1.png'}"
    )

    print(
        f"\n  IoU vs area:\n"
        f"  {OUTPUT_DIR / 'change_area_vs_iou.png'}"
    )

    print(
        "\nEverything completed successfully. 🎉"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()