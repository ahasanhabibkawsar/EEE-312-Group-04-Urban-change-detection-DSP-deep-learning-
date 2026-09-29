"""
============================================================
URBAN CHANGE DETECTION
INFERENCE / PREDICTOR
============================================================

This module handles:

    1. Loading the trained model
    2. Loading two satellite images
    3. Preparing the images
    4. Running inference
    5. Generating probability map
    6. Generating binary change mask
    7. Calculating changed area

============================================================
"""

import os

import cv2
import numpy as np
import torch

from models.siamese_unet import SiameseUNet


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 256

THRESHOLD = 0.5


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
# MODEL LOADING
# ============================================================

def load_model(
    checkpoint_path
):
    """
    Load the trained Siamese U-Net model.
    """

    device = get_device()

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = SiameseUNet()

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # --------------------------------------------------------
    # Handle checkpoint format
    # --------------------------------------------------------

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    print(
        "Model loaded successfully."
    )

    return model, device


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(
    image_path
):
    """
    Load a satellite image using OpenCV.
    """

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )

    image = cv2.imread(
        image_path,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:

        raise ValueError(
            f"Unable to read image:\n"
            f"{image_path}"
        )

    return image


# ============================================================
# IMAGE RESIZING
# ============================================================

def resize_image(
    image,
    size=IMAGE_SIZE
):
    """
    Resize image to model input resolution.
    """

    image = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_LINEAR
    )

    return image


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

def normalize_image(
    image
):
    """
    Convert image into floating point format
    in the range [0, 1].
    """

    image = image.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Handle 8-bit images
    # --------------------------------------------------------

    if image.max() > 1.0:

        image = image / 255.0

    return image


# ============================================================
# CHANNEL PREPARATION
# ============================================================

def prepare_channels(
    image
):
    """
    Convert image into the 4-channel format expected
    by the Siamese U-Net.

    Current model expects:

        [B, 4, H, W]

    For normal RGB images, an additional channel is
    generated from grayscale intensity.
    """

    # --------------------------------------------------------
    # Grayscale image
    # --------------------------------------------------------

    if image.ndim == 2:

        image = np.stack(
            [
                image,
                image,
                image
            ],
            axis=-1
        )

    # --------------------------------------------------------
    # RGB image
    # --------------------------------------------------------

    if image.shape[2] == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = gray[..., np.newaxis]

        image = np.concatenate(
            [
                image,
                gray
            ],
            axis=2
        )

    # --------------------------------------------------------
    # RGBA image
    # --------------------------------------------------------

    elif image.shape[2] == 4:

        pass

    else:

        raise ValueError(
            "Unsupported image channel count: "
            f"{image.shape[2]}"
        )

    return image


# ============================================================
# IMAGE → TENSOR
# ============================================================

def image_to_tensor(
    image
):
    """
    Convert image from:

        H × W × C

    to:

        1 × C × H × W
    """

    image = normalize_image(
        image
    )

    image = prepare_channels(
        image
    )

    image = torch.from_numpy(
        image
    )

    image = image.permute(
        2,
        0,
        1
    )

    image = image.unsqueeze(
        0
    )

    image = image.float()

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    before_image,
    after_image,
    device,
    threshold=THRESHOLD
):
    """
    Run change detection.

    Returns:

        probability_map
        binary_mask
    """

    # --------------------------------------------------------
    # Convert images
    # --------------------------------------------------------

    before_tensor = image_to_tensor(
        before_image
    )

    after_tensor = image_to_tensor(
        after_image
    )

    # --------------------------------------------------------
    # Move tensors to device
    # --------------------------------------------------------

    before_tensor = before_tensor.to(
        device
    )

    after_tensor = after_tensor.to(
        device
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            before_tensor,
            after_tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

    # --------------------------------------------------------
    # Move to CPU
    # --------------------------------------------------------

    probabilities = probabilities.cpu()

    # --------------------------------------------------------
    # Binary mask
    # --------------------------------------------------------

    binary_mask = (
        probabilities >= threshold
    ).float()

    # --------------------------------------------------------
    # Remove batch/channel dimensions
    # --------------------------------------------------------

    probability_map = (
        probabilities[0, 0]
        .numpy()
    )

    binary_mask = (
        binary_mask[0, 0]
        .numpy()
    )

    return (
        probability_map,
        binary_mask
    )


# ============================================================
# CHANGE AREA
# ============================================================

def calculate_change_area(
    binary_mask
):
    """
    Calculate percentage of pixels classified
    as changed.
    """

    total_pixels = binary_mask.size

    changed_pixels = np.sum(
        binary_mask > 0
    )

    percentage = (
        changed_pixels
        / total_pixels
        * 100.0
    )

    return (
        changed_pixels,
        total_pixels,
        percentage
    )


# ============================================================
# COMPLETE INFERENCE FUNCTION
# ============================================================

def run_inference(
    model,
    before_path,
    after_path,
    device,
    threshold=THRESHOLD
):
    """
    Complete inference pipeline.

    Returns a dictionary containing:

        before image
        after image
        probability map
        binary mask
        changed pixels
        total pixels
        changed area percentage
    """

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    before_image = load_image(
        before_path
    )

    after_image = load_image(
        after_path
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    before_image = resize_image(
        before_image
    )

    after_image = resize_image(
        after_image
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability_map, binary_mask = predict(
        model=model,
        before_image=before_image,
        after_image=after_image,
        device=device,
        threshold=threshold
    )

    # --------------------------------------------------------
    # Calculate area
    # --------------------------------------------------------

    changed_pixels, total_pixels, percentage = (
        calculate_change_area(
            binary_mask
        )
    )

    # --------------------------------------------------------
    # Return everything
    # --------------------------------------------------------

    return {
        "before": before_image,
        "after": after_image,
        "probability": probability_map,
        "mask": binary_mask,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_percentage": percentage
    }