"""
DSP-based feature extraction for satellite imagery.

Pipeline (per image, per date):

    RGB (uint8)
     ↓
    Grayscale            Gray = 0.299 R + 0.587 G + 0.114 B
     ↓
    Gaussian low-pass    5 x 5 kernel, sigma = 1.0  (noise suppression)
     ↓
    Canny edge detector  hysteresis thresholds 50 / 150
     ↓
    RGB + Edge  ->  per-channel standardisation  ->  network input

v2 fix: v1 fed RGB in [0, 255] next to an edge map in [0, 1], so the
edge channel was ~255x weaker than the colour channels and the network
effectively ignored it. `prepare_model_input` now standardises every
channel (ImageNet statistics for RGB, measured statistics for the edge
map) so that all four channels have comparable magnitude.
"""

import cv2
import numpy as np

from config import (
    GAUSSIAN_KERNEL_SIZE,
    GAUSSIAN_SIGMA,
    CANNY_LOW_THRESHOLD,
    CANNY_HIGH_THRESHOLD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    EDGE_MEAN,
    EDGE_STD,
    USE_CANNY,
)


_RGB_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_RGB_STD = np.array(IMAGENET_STD, dtype=np.float32)


# =========================================================
# Classical DSP stages
# =========================================================

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale.

    Uses the standard luminance relationship:

    Gray = 0.299R + 0.587G + 0.114B
    """

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_gaussian_filter(
    grayscale,
    kernel_size=GAUSSIAN_KERNEL_SIZE,
    sigma=GAUSSIAN_SIGMA,
):
    """
    Apply Gaussian low-pass filtering (2-D convolution with a
    separable Gaussian kernel).
    """

    return cv2.GaussianBlur(
        grayscale,
        (kernel_size, kernel_size),
        sigmaX=sigma,
    )


def detect_canny_edges(
    blurred,
    low_threshold=CANNY_LOW_THRESHOLD,
    high_threshold=CANNY_HIGH_THRESHOLD,
):
    """
    Detect edges using the Canny algorithm
    (Sobel gradients -> non-maximum suppression -> hysteresis).

    Returns an edge image with values 0 or 255.
    """

    return cv2.Canny(
        blurred,
        threshold1=low_threshold,
        threshold2=high_threshold,
    )


def normalize_edge_map(edges):
    """
    Convert Canny output from {0, 255} to {0, 1}.
    """

    return edges.astype(np.float32) / 255.0


def compute_edge_map(rgb_image):
    """
    Grayscale -> Gaussian -> Canny.

    Returns a float32 H x W edge map with values in {0, 1}.
    """

    grayscale = rgb_to_grayscale(rgb_image)
    blurred = apply_gaussian_filter(grayscale)
    edges = detect_canny_edges(blurred)
    return normalize_edge_map(edges)


def create_four_channel_input(rgb_image, edge_map):
    """
    Stack an RGB image (H x W x 3) and an edge map (H x W)
    into an H x W x 4 array (no normalisation).
    """

    edge_channel = edge_map[..., np.newaxis]
    four_channel = np.concatenate([rgb_image, edge_channel], axis=-1)
    return four_channel.astype(np.float32)


def extract_dsp_features(rgb_image):
    """
    Complete DSP feature-extraction pipeline, UN-normalised.

    Kept for visualisation of the DSP stages. The network must be fed
    with `prepare_model_input`, which also standardises the channels.

    Returns
    -------
    four_channel : np.ndarray   H x W x 4  (RGB in [0, 255], edge in {0, 1})
    edge_map     : np.ndarray   H x W
    """

    edges = compute_edge_map(rgb_image)
    four_channel = create_four_channel_input(rgb_image, edges)
    return four_channel, edges


# =========================================================
# Network input
# =========================================================

def normalize_rgb_for_model(rgb_image):
    """uint8 RGB -> float32 RGB standardised with ImageNet statistics."""

    rgb = rgb_image.astype(np.float32) / 255.0
    return (rgb - _RGB_MEAN) / _RGB_STD


def normalize_edge_for_model(edge_map):
    """{0, 1} edge map -> standardised edge channel."""

    return (edge_map.astype(np.float32) - EDGE_MEAN) / EDGE_STD


def prepare_model_input(rgb_image, use_canny=USE_CANNY):
    """
    Build the exact tensor layout the network was trained on.

    Parameters
    ----------
    rgb_image : np.ndarray  uint8, H x W x 3
    use_canny : bool        append the standardised Canny channel

    Returns
    -------
    np.ndarray  float32, H x W x 4  (or H x W x 3 when use_canny=False)
    """

    rgb = normalize_rgb_for_model(rgb_image)

    if not use_canny:
        return np.ascontiguousarray(rgb, dtype=np.float32)

    edge = normalize_edge_for_model(compute_edge_map(rgb_image))
    return np.ascontiguousarray(
        np.concatenate([rgb, edge[..., None]], axis=-1),
        dtype=np.float32,
    )


def denormalize_for_display(image):
    """
    Undo `prepare_model_input` for plotting.

    Accepts a torch tensor or numpy array shaped C x H x W, 1 x C x H x W
    or H x W x C (C >= 3). Returns an H x W x 3 float array in [0, 1].
    """

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    image = np.asarray(image, dtype=np.float32)

    if image.ndim == 4:
        image = image[0]

    # Channel-first -> channel-last
    if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.transpose(image, (1, 2, 0))

    rgb = image[..., :3] * _RGB_STD + _RGB_MEAN
    return np.clip(rgb, 0.0, 1.0)
