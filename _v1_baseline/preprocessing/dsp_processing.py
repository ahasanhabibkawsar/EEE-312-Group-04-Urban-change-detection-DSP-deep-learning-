"""
DSP-based feature extraction for satellite imagery.

Pipeline:

RGB
 ↓
Grayscale
 ↓
Gaussian filtering
 ↓
Canny edge detection
 ↓
RGB + Edge
"""

import cv2
import numpy as np

from config import (
    GAUSSIAN_KERNEL_SIZE,
    GAUSSIAN_SIGMA,
    CANNY_LOW_THRESHOLD,
    CANNY_HIGH_THRESHOLD,
)


def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale.

    Uses the standard luminance relationship:

    Gray = 0.299R + 0.587G + 0.114B
    """

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    return grayscale


def apply_gaussian_filter(
    grayscale,
    kernel_size=GAUSSIAN_KERNEL_SIZE,
    sigma=GAUSSIAN_SIGMA
):
    """
    Apply Gaussian low-pass filtering.

    Parameters
    ----------
    grayscale : np.ndarray
        Grayscale image.

    kernel_size : int
        Size of Gaussian kernel.

    sigma : float
        Gaussian standard deviation.

    Returns
    -------
    np.ndarray
        Smoothed grayscale image.
    """

    blurred = cv2.GaussianBlur(
        grayscale,
        (kernel_size, kernel_size),
        sigmaX=sigma
    )

    return blurred


def detect_canny_edges(
    blurred,
    low_threshold=CANNY_LOW_THRESHOLD,
    high_threshold=CANNY_HIGH_THRESHOLD
):
    """
    Detect edges using the Canny algorithm.

    Returns an edge image with values 0 or 255.
    """

    edges = cv2.Canny(
        blurred,
        threshold1=low_threshold,
        threshold2=high_threshold
    )

    return edges


def normalize_edge_map(edges):
    """
    Convert Canny output from [0,255] to [0,1].
    """

    return edges.astype(np.float32) / 255.0


def create_four_channel_input(
    rgb_image,
    edge_map
):
    """
    Combine RGB image and edge map.

    Input:
        RGB   -> H x W x 3
        Edge  -> H x W

    Output:
        RGB + Edge -> H x W x 4
    """

    edge_channel = edge_map[..., np.newaxis]

    four_channel = np.concatenate(
        [rgb_image, edge_channel],
        axis=-1
    )

    return four_channel.astype(np.float32)


def extract_dsp_features(rgb_image):
    """
    Complete DSP feature-extraction pipeline.

    RGB
     ↓
    Grayscale
     ↓
    Gaussian
     ↓
    Canny
     ↓
    RGB + Edge

    Returns
    -------
    four_channel : np.ndarray
        H x W x 4

    edge_map : np.ndarray
        H x W
    """

    grayscale = rgb_to_grayscale(rgb_image)

    blurred = apply_gaussian_filter(
        grayscale
    )

    edges = detect_canny_edges(
        blurred
    )

    edges = normalize_edge_map(
        edges
    )

    four_channel = create_four_channel_input(
        rgb_image,
        edges
    )

    return four_channel, edges