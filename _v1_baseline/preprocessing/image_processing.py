"""
Basic image loading and preprocessing utilities.
Includes Sobel edge detection for ground-truth edge supervision.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def load_rgb_image(image_path):
    """
    Load an image from disk and convert BGR to RGB.

    Parameters
    ----------
    image_path : str or Path
        Path to the image.

    Returns
    -------
    np.ndarray
        RGB image with shape (H, W, 3).
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_mask(mask_path):
    """
    Load a change-detection ground-truth mask.

    Parameters
    ----------
    mask_path : str or Path
        Path to mask.

    Returns
    -------
    np.ndarray
        Grayscale mask.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")
    return mask


def resize_image(image, size):
    """
    Resize an RGB image.

    Bilinear interpolation is used for satellite images.
    """
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def resize_mask(mask, size):
    """
    Resize a segmentation mask.

    Nearest-neighbor interpolation is used so that
    class labels are not interpolated.
    """
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def normalize_rgb(image):
    """
    Convert RGB pixel values from [0, 255] to [0, 1].
    """
    image = image.astype(np.float32)
    return image / 255.0


def normalize_mask(mask):
    """
    Convert a binary mask to values 0 or 1.
    """
    mask = (mask > 127).astype(np.float32)
    return mask


def sobel_edge_detection(mask_tensor):
    """
    Computes binary edges from a binary mask tensor using Sobel operator.

    Input: mask_tensor [B, 1, H, W] or [1, H, W]
    Output: binary edge tensor with same shape [B, 1, H, W]
    """
    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.unsqueeze(0)
        
    # Define Sobel kernels (float)
    kernel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
        dtype=torch.float32
    ).view(1, 1, 3, 3).to(mask_tensor.device)
    
    kernel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
        dtype=torch.float32
    ).view(1, 1, 3, 3).to(mask_tensor.device)
    
    # Convolve
    gx = F.conv2d(mask_tensor, kernel_x, padding=1)
    gy = F.conv2d(mask_tensor, kernel_y, padding=1)
    
    # Magnitude
    grad_mag = torch.sqrt(gx**2 + gy**2)
    
    # Threshold to binary (values > 0.5 are considered edges)
    edges = (grad_mag > 0.5).float()
    
    return edges