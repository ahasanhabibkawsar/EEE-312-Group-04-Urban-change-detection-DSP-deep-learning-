"""
============================================================
URBAN CHANGE DETECTION
SINGLE IMAGE / TEST SET INFERENCE
============================================================

Project:
    Urban Change Detection using Siamese U-Net

Dataset:
    LEVIR-CD

Directory structure:

    Urban_Change_Detection/
    │
    ├── data/
    │   └── LEVIR-CD/
    │       ├── train/
    │       │   ├── A/
    │       │   ├── B/
    │       │   └── label/
    │       │
    │       ├── val/
    │       │   ├── A/
    │       │   ├── B/
    │       │   └── label/
    │       │
    │       └── test/
    │           ├── A/
    │           ├── B/
    │           └── label/
    │
    ├── checkpoints/
    │   ├── best_model.pth
    │   └── latest_model.pth
    │
    ├── dataset/
    │   └── levir_dataset.py
    │
    ├── models/
    │   └── siamese_unet.py
    │
    └── test_inference.py

============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset.levir_dataset import LEVIRCDDataset
from models.siamese_unet import SiameseUNet


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 256

# Threshold for converting probability into binary prediction
THRESHOLD = 0.5

# Which test sample should be visualized?
# 0 = first test image
SAMPLE_INDEX = 0

# Set this to True if you want to evaluate all test images
RUN_FULL_TEST = False

# Maximum number of visualization samples when RUN_FULL_TEST=True
MAX_VISUALIZATIONS = 10


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# IMPORTANT:
# Your dataset is inside "data", NOT "dataset".
DATASET_ROOT = PROJECT_ROOT / "data" / "LEVIR-CD"

# Test directories
TEST_A_DIR = DATASET_ROOT / "test" / "A"
TEST_B_DIR = DATASET_ROOT / "test" / "B"
TEST_LABEL_DIR = DATASET_ROOT / "test" / "label"

# Trained model
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

# Output directory
RESULTS_DIR = PROJECT_ROOT / "inference_results"


# ============================================================
# DEVICE
# ============================================================

def get_device():
    """
    Select the best available device.

    Priority:
        1. Apple MPS
        2. CUDA
        3. CPU
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# PRINT HEADER
# ============================================================

def print_header(title):
    """
    Print a formatted section header.
    """

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def check_required_files():
    """
    Check whether all required directories and files exist.
    """

    print_header("CHECKING REQUIRED FILES")

    print(f"\nDataset root : {DATASET_ROOT}")
    print(f"Test A       : {TEST_A_DIR}")
    print(f"Test B       : {TEST_B_DIR}")
    print(f"Test labels  : {TEST_LABEL_DIR}")
    print(f"Checkpoint   : {CHECKPOINT_PATH}")

    # Dataset root
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"\nDataset root not found:\n{DATASET_ROOT}\n\n"
            f"Expected structure:\n"
            f"{DATASET_ROOT}/train/\n"
            f"{DATASET_ROOT}/val/\n"
            f"{DATASET_ROOT}/test/"
        )

    # Test A
    if not TEST_A_DIR.exists():
        raise FileNotFoundError(
            f"\nTest A directory not found:\n{TEST_A_DIR}"
        )

    # Test B
    if not TEST_B_DIR.exists():
        raise FileNotFoundError(
            f"\nTest B directory not found:\n{TEST_B_DIR}"
        )

    # Test labels
    if not TEST_LABEL_DIR.exists():
        raise FileNotFoundError(
            f"\nTest label directory not found:\n{TEST_LABEL_DIR}"
        )

    # Checkpoint
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"\nModel checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    print("\nAll required files found. ✅")


# ============================================================
# GET TEST FILENAMES
# ============================================================

def get_test_filenames():
    """
    Find filenames that exist in A, B and label directories.
    """

    valid_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }

    files_a = {
        file.name
        for file in TEST_A_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in valid_extensions
    }

    files_b = {
        file.name
        for file in TEST_B_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in valid_extensions
    }

    files_label = {
        file.name
        for file in TEST_LABEL_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in valid_extensions
    }

    filenames = sorted(
        list(files_a.intersection(files_b).intersection(files_label))
    )

    if len(filenames) == 0:
        raise RuntimeError(
            "No matching test samples were found.\n\n"
            "Make sure filenames in A, B and label match."
        )

    return filenames


# ============================================================
# CREATE TEST DATASET
# ============================================================

def create_test_dataset():
    """
    Create the LEVIR-CD test dataset.

    This matches the actual constructor of LEVIRCDDataset:

        LEVIRCDDataset(
            image_a_dir,
            image_b_dir,
            label_dir,
            image_size,
            file_names,
            augment
        )
    """

    print_header("CREATING TEST DATASET")

    filenames = get_test_filenames()

    print(f"Valid test samples: {len(filenames)}")

    dataset = LEVIRCDDataset(
        image_a_dir=TEST_A_DIR,
        image_b_dir=TEST_B_DIR,
        label_dir=TEST_LABEL_DIR,
        image_size=IMAGE_SIZE,
        file_names=filenames,
        augment=False,
    )

    print(f"Dataset samples   : {len(dataset)}")

    return dataset, filenames


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(device):
    """
    Create the Siamese U-Net model.
    """

    print_header("CREATING SIAMESE U-NET")

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

    print(f"Total parameters    : {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model, device):
    """
    Load the best trained model checkpoint.
    """

    print_header("LOADING TRAINED MODEL")

    print(f"Checkpoint:\n{CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    # --------------------------------------------------------
    # Different possible checkpoint formats are supported.
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            # Sometimes the checkpoint itself is the state dict.
            state_dict = checkpoint

    else:

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    # --------------------------------------------------------
    # Remove possible DataParallel prefix.
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            new_key = key[len("module."):]

        else:

            new_key = key

        cleaned_state_dict[new_key] = value

    # --------------------------------------------------------
    # Load weights.
    # --------------------------------------------------------

    missing_keys, unexpected_keys = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if len(missing_keys) > 0:

        print("\nWarning: Missing keys:")

        for key in missing_keys:
            print("  ", key)

    if len(unexpected_keys) > 0:

        print("\nWarning: Unexpected keys:")

        for key in unexpected_keys:
            print("  ", key)

    # --------------------------------------------------------
    # Print checkpoint information.
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:
            print(f"\nCheckpoint epoch: {checkpoint['epoch']}")

        if "val_f1" in checkpoint:
            print(
                f"Validation F1: "
                f"{checkpoint['val_f1']:.6f}"
            )

        elif "best_val_f1" in checkpoint:
            print(
                f"Validation F1: "
                f"{checkpoint['best_val_f1']:.6f}"
            )

    print("\nModel loaded successfully. ✅")

    return model


# ============================================================
# PREPARE IMAGE FOR DISPLAY
# ============================================================

def prepare_rgb_image(image):
    """
    Convert a tensor image into an RGB numpy image.

    LEVIR-CD images contain 4 channels in this project.

    We use the first 3 channels for visualization.
    """

    if isinstance(image, torch.Tensor):

        image = image.detach().cpu()

    # Remove batch dimension if present
    if image.ndim == 4:
        image = image[0]

    # CHW -> HWC
    if image.ndim == 3:

        image = image.permute(1, 2, 0)

    image = image.numpy()

    # Use first three channels for RGB visualization
    if image.shape[-1] >= 3:

        image = image[:, :, :3]

    # Normalize for display
    image_min = image.min()
    image_max = image.max()

    if image_max > image_min:

        image = (
            image - image_min
        ) / (
            image_max - image_min
        )

    else:

        image = np.zeros_like(image)

    return np.clip(image, 0, 1)


# ============================================================
# PREPARE MASK
# ============================================================

def prepare_mask(mask):
    """
    Convert mask tensor into 2D numpy array.
    """

    if isinstance(mask, torch.Tensor):

        mask = mask.detach().cpu()

    # Remove batch dimension
    if mask.ndim == 4:
        mask = mask[0]

    # Remove channel dimension
    if mask.ndim == 3:
        mask = mask[0]

    mask = mask.numpy()

    return (mask > 0.5).astype(np.uint8)


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(prediction, target):
    """
    Calculate binary segmentation metrics.
    """

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

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    total = tp + tn + fp + fn

    if total > 0:
        accuracy = (tp + tn) / total
    else:
        accuracy = 0.0

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if precision + recall > 0:

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    if tp + fp + fn > 0:

        iou = (
            tp
            / (tp + fp + fn)
        )

    else:

        iou = 0.0

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
# RUN SINGLE INFERENCE
# ============================================================

@torch.no_grad()
def run_single_inference(
    model,
    dataset,
    index,
    device,
):
    """
    Run inference on one test image.
    """

    if index < 0 or index >= len(dataset):

        raise IndexError(
            f"Sample index {index} is out of range.\n"
            f"Valid range: 0 - {len(dataset) - 1}"
        )

    sample = dataset[index]

    # --------------------------------------------------------
    # Dataset returns dictionary:
    #
    # before
    # after
    # mask
    # filename
    # --------------------------------------------------------

    before = sample["before"]
    after = sample["after"]
    target = sample["mask"]

    filename = sample.get(
        "filename",
        f"sample_{index}.png"
    )

    # --------------------------------------------------------
    # Add batch dimension.
    # --------------------------------------------------------

    before_input = before.unsqueeze(0).to(device)
    after_input = after.unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Model inference.
    # --------------------------------------------------------

    logits = model(
        before_input,
        after_input,
    )

    # --------------------------------------------------------
    # Convert logits to probability.
    # --------------------------------------------------------

    probability = torch.sigmoid(logits)

    # --------------------------------------------------------
    # Binary prediction.
    # --------------------------------------------------------

    prediction = (
        probability >= THRESHOLD
    ).float()

    # --------------------------------------------------------
    # Remove batch/channel where appropriate.
    # --------------------------------------------------------

    probability_np = (
        probability[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    prediction_np = (
        prediction[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    target_np = prepare_mask(target)

    metrics = calculate_metrics(
        prediction_np,
        target_np,
    )

    return {
        "before": before,
        "after": after,
        "target": target_np,
        "probability": probability_np,
        "prediction": prediction_np,
        "filename": filename,
        "metrics": metrics,
    }


# ============================================================
# PRINT SINGLE IMAGE RESULTS
# ============================================================

def print_single_results(result):
    """
    Print metrics and prediction statistics.
    """

    metrics = result["metrics"]

    probability = result["probability"]
    prediction = result["prediction"]
    target = result["target"]

    print_header("PREDICTION RESULTS")

    print(f"\nFilename: {result['filename']}")

    print("\nProbability:")
    print(
        f"Minimum : {probability.min():.6f}"
    )
    print(
        f"Maximum : {probability.max():.6f}"
    )
    print(
        f"Mean    : {probability.mean():.6f}"
    )

    predicted_positive = int(
        prediction.sum()
    )

    ground_truth_positive = int(
        target.sum()
    )

    total_pixels = prediction.size

    print("\nPixel statistics:")

    print(
        f"Total pixels           : "
        f"{total_pixels:,}"
    )

    print(
        f"Predicted change pixels: "
        f"{predicted_positive:,}"
    )

    print(
        f"Ground truth pixels    : "
        f"{ground_truth_positive:,}"
    )

    print(
        f"Predicted change %     : "
        f"{100 * predicted_positive / total_pixels:.4f}%"
    )

    print(
        f"Ground truth change %   : "
        f"{100 * ground_truth_positive / total_pixels:.4f}%"
    )

    print("\nMetrics:")

    print(
        f"Accuracy  : "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1-score  : "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"IoU       : "
        f"{metrics['iou']:.6f}"
    )

    print("\nConfusion Matrix:")

    print(
        f"TP: {metrics['tp']}"
    )

    print(
        f"TN: {metrics['tn']}"
    )

    print(
        f"FP: {metrics['fp']}"
    )

    print(
        f"FN: {metrics['fn']}"
    )


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(result, index):
    """
    Save:

        Before image
        After image
        Ground Truth
        Prediction
        Probability Map
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    before_image = prepare_rgb_image(
        result["before"]
    )

    after_image = prepare_rgb_image(
        result["after"]
    )

    target = result["target"]

    prediction = result["prediction"]

    probability = result["probability"]

    filename = result["filename"]

    # Remove extension
    stem = Path(filename).stem

    output_path = (
        RESULTS_DIR
        / f"{index:03d}_{stem}_inference.png"
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    # --------------------------------------------------------
    # Before
    # --------------------------------------------------------

    axes[0, 0].imshow(
        before_image
    )

    axes[0, 0].set_title(
        "Before Image"
    )

    axes[0, 0].axis("off")

    # --------------------------------------------------------
    # After
    # --------------------------------------------------------

    axes[0, 1].imshow(
        after_image
    )

    axes[0, 1].set_title(
        "After Image"
    )

    axes[0, 1].axis("off")

    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    axes[0, 2].imshow(
        target,
        cmap="gray"
    )

    axes[0, 2].set_title(
        "Ground Truth"
    )

    axes[0, 2].axis("off")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    axes[1, 0].imshow(
        prediction,
        cmap="gray"
    )

    axes[1, 0].set_title(
        f"Predicted Change\n"
        f"Threshold = {THRESHOLD}"
    )

    axes[1, 0].axis("off")

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    axes[1, 1].imshow(
        probability,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    axes[1, 1].set_title(
        "Change Probability"
    )

    axes[1, 1].axis("off")

    # --------------------------------------------------------
    # Difference / Overlay
    # --------------------------------------------------------

    overlay = before_image.copy()

    # Red-like visualization through RGB channels
    changed_pixels = prediction > 0.5

    overlay[changed_pixels] = [
        1.0,
        0.0,
        0.0
    ]

    axes[1, 2].imshow(
        overlay
    )

    axes[1, 2].set_title(
        "Detected Changes"
    )

    axes[1, 2].axis("off")

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    metrics = result["metrics"]

    fig.suptitle(
        f"Urban Change Detection\n"
        f"{filename} | "
        f"F1 = {metrics['f1']:.4f} | "
        f"IoU = {metrics['iou']:.4f}",
        fontsize=14
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"\nVisualization saved:\n"
        f"{output_path}"
    )

    return output_path


# ============================================================
# RUN FULL TEST SET
# ============================================================

@torch.no_grad()
def run_full_test(
    model,
    dataset,
    device,
):
    """
    Run inference on the entire test dataset.
    """

    print_header(
        "FULL TEST SET INFERENCE"
    )

    all_metrics = []

    number_of_samples = len(dataset)

    print(
        f"Test samples: "
        f"{number_of_samples}"
    )

    for index in range(number_of_samples):

        print(
            f"\nProcessing "
            f"{index + 1}/{number_of_samples}"
        )

        result = run_single_inference(
            model=model,
            dataset=dataset,
            index=index,
            device=device,
        )

        all_metrics.append(
            result["metrics"]
        )

        # Save only a limited number of visualizations
        if index < MAX_VISUALIZATIONS:

            save_visualization(
                result,
                index,
            )

    # --------------------------------------------------------
    # Calculate average metrics
    # --------------------------------------------------------

    if len(all_metrics) == 0:

        return

    accuracy = np.mean(
        [
            m["accuracy"]
            for m in all_metrics
        ]
    )

    precision = np.mean(
        [
            m["precision"]
            for m in all_metrics
        ]
    )

    recall = np.mean(
        [
            m["recall"]
            for m in all_metrics
        ]
    )

    f1 = np.mean(
        [
            m["f1"]
            for m in all_metrics
        ]
    )

    iou = np.mean(
        [
            m["iou"]
            for m in all_metrics
        ]
    )

    print_header(
        "AVERAGE TEST METRICS"
    )

    print(
        f"Accuracy  : {accuracy:.6f}"
    )

    print(
        f"Precision : {precision:.6f}"
    )

    print(
        f"Recall    : {recall:.6f}"
    )

    print(
        f"F1-score  : {f1:.6f}"
    )

    print(
        f"IoU       : {iou:.6f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "URBAN CHANGE DETECTION\n"
        "SINGLE IMAGE INFERENCE"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    check_required_files()

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    dataset, filenames = (
        create_test_dataset()
    )

    # --------------------------------------------------------
    # Print available samples
    # --------------------------------------------------------

    print_header(
        "AVAILABLE TEST SAMPLES"
    )

    print(
        f"Number of samples: "
        f"{len(dataset)}"
    )

    print(
        f"\nSelected sample index: "
        f"{SAMPLE_INDEX}"
    )

    print(
        f"Selected filename: "
        f"{filenames[SAMPLE_INDEX]}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model(
        device
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model = load_checkpoint(
        model,
        device
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    print(
        "\nModel switched to evaluation mode. ✅"
    )

    # --------------------------------------------------------
    # Full test set
    # --------------------------------------------------------

    if RUN_FULL_TEST:

        run_full_test(
            model=model,
            dataset=dataset,
            device=device,
        )

    # --------------------------------------------------------
    # Single image
    # --------------------------------------------------------

    else:

        print_header(
            "RUNNING SINGLE IMAGE INFERENCE"
        )

        result = run_single_inference(
            model=model,
            dataset=dataset,
            index=SAMPLE_INDEX,
            device=device,
        )

        print_single_results(
            result
        )

        save_visualization(
            result,
            SAMPLE_INDEX,
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print_header(
        "INFERENCE COMPLETE"
    )

    print(
        f"\nResults directory:\n"
        f"{RESULTS_DIR}"
    )

    print(
        "\nEverything completed successfully. ✅"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()