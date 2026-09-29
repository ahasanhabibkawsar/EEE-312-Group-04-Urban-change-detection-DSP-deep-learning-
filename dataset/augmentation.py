"""
Augmentation for bi-temporal change-detection images.

Two families:

1. Geometric (SYNCHRONISED)
   The same flip / rotation / crop is applied to Before, After and the
   ground-truth mask, so the three stay pixel-aligned.

2. Photometric (INDEPENDENT per date)
   Brightness, contrast and colour are jittered separately for the
   Before and After image. Real image pairs differ in season, sun angle
   and sensor; this teaches the network that such differences are
   NOT "change". The mask is never touched.

All functions work on uint8 H x W x C numpy arrays and run BEFORE the
DSP stage, so the Canny map is computed on the augmented image.
"""

import random

import numpy as np


# =========================================================
# Geometric (synchronised)
# =========================================================

def random_horizontal_flip(before, after, mask, probability=0.5):
    """Randomly flip all inputs horizontally."""

    if random.random() < probability:
        before = np.fliplr(before).copy()
        after = np.fliplr(after).copy()
        mask = np.fliplr(mask).copy()

    return before, after, mask


def random_vertical_flip(before, after, mask, probability=0.5):
    """Randomly flip all inputs vertically."""

    if random.random() < probability:
        before = np.flipud(before).copy()
        after = np.flipud(after).copy()
        mask = np.flipud(mask).copy()

    return before, after, mask


def random_rotation(before, after, mask, probability=0.5):
    """Randomly rotate all inputs by 90, 180 or 270 degrees."""

    if random.random() < probability:
        k = random.choice([1, 2, 3])
        before = np.rot90(before, k).copy()
        after = np.rot90(after, k).copy()
        mask = np.rot90(mask, k).copy()

    return before, after, mask


def random_crop(before, after, mask, crop_size, change_focus_prob=0.0):
    """
    Cut the same crop_size x crop_size window out of all three arrays.

    With probability `change_focus_prob` the window is placed so that it
    contains a randomly chosen changed pixel (at a random position inside
    the window). This counteracts the strong class imbalance of LEVIR-CD
    (~4.6 % changed pixels, 18 % of images contain no change at all).
    """

    height, width = mask.shape[:2]

    if height < crop_size or width < crop_size:
        raise ValueError(
            f"Image {height}x{width} is smaller than crop size {crop_size}"
        )

    max_y = height - crop_size
    max_x = width - crop_size

    y0 = x0 = None

    if change_focus_prob > 0 and random.random() < change_focus_prob:
        ys, xs = np.nonzero(mask > 127 if mask.dtype == np.uint8 else mask > 0.5)

        if len(ys) > 0:
            i = random.randrange(len(ys))
            cy, cx = int(ys[i]), int(xs[i])
            y0 = min(max(cy - random.randrange(crop_size), 0), max_y)
            x0 = min(max(cx - random.randrange(crop_size), 0), max_x)

    if y0 is None:
        y0 = random.randint(0, max_y)
        x0 = random.randint(0, max_x)

    window = (slice(y0, y0 + crop_size), slice(x0, x0 + crop_size))

    return (
        before[window].copy(),
        after[window].copy(),
        mask[window].copy(),
    )


# =========================================================
# Photometric (independent per date)
# =========================================================

def random_photometric(image, probability=0.5):
    """
    Mild brightness / contrast / per-channel colour jitter on ONE image.

        out = contrast * (img - mean) + mean + brightness + colour_shift
    """

    if random.random() >= probability:
        return image

    img = image.astype(np.float32)

    contrast = random.uniform(0.8, 1.2)
    brightness = random.uniform(-20.0, 20.0)
    colour_shift = np.random.uniform(-8.0, 8.0, size=(1, 1, img.shape[2]))

    mean = img.mean(axis=(0, 1), keepdims=True)
    img = contrast * (img - mean) + mean + brightness + colour_shift

    return np.clip(img, 0, 255).astype(np.uint8)


# =========================================================
# Full training augmentation
# =========================================================

def apply_training_augmentation(before, after, mask, photometric=True):
    """
    Apply the training augmentation.

    Geometric transforms are shared by Before, After and Mask;
    photometric transforms are drawn independently for each date.
    """

    before, after, mask = random_horizontal_flip(before, after, mask)
    before, after, mask = random_vertical_flip(before, after, mask)
    before, after, mask = random_rotation(before, after, mask)

    if photometric:
        before = random_photometric(before)
        after = random_photometric(after)

    return before, after, mask
