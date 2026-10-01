"""
Synchronized augmentation for change-detection images.

The same spatial transformation must be applied to:

    Before image
    After image
    Ground-truth mask

This keeps all three images spatially aligned.
"""

import random

import numpy as np
import torch


def random_horizontal_flip(
    before,
    after,
    mask,
    probability=0.5,
):
    """
    Randomly flip all inputs horizontally.

    Parameters
    ----------
    before : np.ndarray
        Before image.

    after : np.ndarray
        After image.

    mask : np.ndarray
        Ground-truth mask.

    probability : float
        Probability of applying the flip.

    Returns
    -------
    before, after, mask
    """

    if random.random() < probability:

        before = np.fliplr(before).copy()
        after = np.fliplr(after).copy()
        mask = np.fliplr(mask).copy()

    return before, after, mask


def random_vertical_flip(
    before,
    after,
    mask,
    probability=0.5,
):
    """
    Randomly flip all inputs vertically.
    """

    if random.random() < probability:

        before = np.flipud(before).copy()
        after = np.flipud(after).copy()
        mask = np.flipud(mask).copy()

    return before, after, mask


def random_rotation(
    before,
    after,
    mask,
    probability=0.5,
):
    """
    Randomly rotate all inputs by 90, 180, or 270 degrees.
    """

    if random.random() < probability:

        rotation = random.choice(
            [1, 2, 3]
        )

        before = np.rot90(
            before,
            rotation
        ).copy()

        after = np.rot90(
            after,
            rotation
        ).copy()

        mask = np.rot90(
            mask,
            rotation
        ).copy()

    return before, after, mask


def apply_training_augmentation(
    before,
    after,
    mask,
):
    """
    Apply synchronized training augmentation.

    All transformations are applied identically
    to Before, After and Mask.
    """

    before, after, mask = random_horizontal_flip(
        before,
        after,
        mask,
    )

    before, after, mask = random_vertical_flip(
        before,
        after,
        mask,
    )

    before, after, mask = random_rotation(
        before,
        after,
        mask,
    )

    return before, after, mask