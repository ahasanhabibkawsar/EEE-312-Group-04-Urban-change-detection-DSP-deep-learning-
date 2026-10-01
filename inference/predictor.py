"""
Inference on arbitrary Before / After image pairs (v2).

Used by the GUI, test_all_inference.py and for new imagery
(e.g. the Purbachal pairs).

    load RGB pair  ->  (optional crop / rescale)  ->  DSP + standardise
        ->  sliding-window prediction at native resolution
        ->  (optional TTA)  ->  probability map  ->  threshold
        ->  (optional post-processing)

Resolution matters: the network was trained at LEVIR-CD's 0.5 m/pixel.
For imagery with a different ground sampling distance (GSD), pass
`scale = source_GSD / 0.5` (e.g. 1.0 m/px imagery -> scale 2.0) so
buildings appear at the size the network learned.
"""

import cv2
import numpy as np
import torch

import config
from models import build_model_from_checkpoint, predict_probabilities
from preprocessing.dsp_processing import prepare_model_input


THRESHOLD = config.load_best_threshold()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device=None, checkpoint_path=config.CHECKPOINT_PATH):
    """Build the model from a v2 checkpoint (in_channels read from it)."""

    device = device or get_device()
    model, _ = build_model_from_checkpoint(checkpoint_path, device)
    return model


def load_image(path):
    """Load an image file as uint8 RGB (H x W x 3)."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def center_crop(image, size):
    h, w = image.shape[:2]
    size = min(size, h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return image[y0:y0 + size, x0:x0 + size]


def rescale(image, scale, is_mask=False):
    """Resample an image by `scale` (>1 enlarges)."""

    if scale == 1.0:
        return image
    h, w = image.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if is_mask:
        return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, size, interpolation=interp)


def match_sizes(before, after):
    """If the two dates differ in size, resize After onto Before's grid."""

    if before.shape[:2] != after.shape[:2]:
        after = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_LINEAR)
    return before, after


def to_tensor(rgb, use_canny, device):
    array = prepare_model_input(rgb, use_canny=use_canny)
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)


@torch.no_grad()
def predict(model, before_image, after_image, device, threshold=THRESHOLD, tta=False):
    """
    Predict a uint8 RGB pair of any size.

    Returns
    -------
    probability_map : float32 H x W in [0, 1]
    binary_mask     : float32 H x W in {0, 1}
    """

    use_canny = model.in_channels == 4
    before_tensor = to_tensor(before_image, use_canny, device)
    after_tensor = to_tensor(after_image, use_canny, device)

    probability = predict_probabilities(model, before_tensor, after_tensor, tta=tta)
    probability_map = probability[0, 0].float().cpu().numpy()

    return probability_map, (probability_map >= threshold).astype(np.float32)


def postprocess_mask(binary_mask, min_area=20, open_kernel=3, polygonize=True, epsilon_ratio=0.01):
    """
    Clean a binary change mask.

    1. Morphological opening (3 x 3) removes isolated false-positive pixels.
    2. Connected regions smaller than `min_area` pixels are dropped.
    3. Optional polygon approximation (Douglas-Peucker, cv2.approxPolyDP)
       straightens outlines into building-like polygons.
    """

    binary = (np.asarray(binary_mask) > 0.5).astype(np.uint8)

    if open_kernel and open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel, open_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(binary)

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        if polygonize:
            epsilon = epsilon_ratio * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(cleaned, [contour], 0, 1, thickness=-1)

    return cleaned.astype(np.float32)


def calculate_change_area(binary_mask):
    """Number / percentage of pixels classified as changed."""

    total_pixels = int(binary_mask.size)
    changed_pixels = int(np.sum(binary_mask > 0))
    return changed_pixels, total_pixels, 100.0 * changed_pixels / max(total_pixels, 1)


def run_inference(model, before_path, after_path, device, threshold=THRESHOLD,
                  tta=False, crop_size=None, scale=1.0):
    """
    Complete inference pipeline for two image files.

    crop_size : optional centre crop (pixels) applied to both images
    scale     : resampling factor (source GSD / 0.5 m)
    """

    before_image = load_image(before_path)
    after_image = load_image(after_path)
    before_image, after_image = match_sizes(before_image, after_image)

    if crop_size:
        before_image = center_crop(before_image, crop_size)
        after_image = center_crop(after_image, crop_size)

    before_image = rescale(before_image, scale)
    after_image = rescale(after_image, scale)

    probability_map, binary_mask = predict(model, before_image, after_image, device, threshold, tta)
    changed_pixels, total_pixels, percentage = calculate_change_area(binary_mask)

    return {
        "before": before_image,
        "after": after_image,
        "probability": probability_map,
        "mask": binary_mask,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_percentage": percentage,
    }
