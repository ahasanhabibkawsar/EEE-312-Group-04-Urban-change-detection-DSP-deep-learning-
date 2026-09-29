"""
LEVIR-CD Dataset

Loads paired satellite images:

    A      -> image before change
    B      -> image after change
    label  -> ground-truth change mask

Three sampling modes
--------------------

"random_crop"  (training, default when augment=True)
    Native-resolution random crops of `image_size` x `image_size`
    pixels. Each image yields `crops_per_image` crops per epoch.

        1024 x 1024 image
            ↓  random (or change-focused) 256 x 256 crop
            ↓  synchronised flips / rotations
            ↓  independent photometric jitter
            ↓  DSP: Gaussian + Canny
            ↓  channel standardisation
        4 x 256 x 256 tensor

"full"  (evaluation, default when augment=False)
    The complete native-resolution image (1024 x 1024 for LEVIR-CD).
    The model predicts it with an overlapping sliding window
    (see SiameseUNet.sliding_window_forward), so no detail is lost.

"resize"  (legacy v1 behaviour)
    Resize the whole image to image_size x image_size.
    Only kept to reproduce the old experiments.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    USE_CANNY,
    CHANGE_FOCUSED_CROP_PROB,
    PHOTOMETRIC_AUGMENTATION,
)

from preprocessing.image_processing import (
    load_rgb_image,
    load_mask,
    resize_image,
    resize_mask,
    normalize_mask,
)

from preprocessing.dsp_processing import prepare_model_input

from dataset.augmentation import (
    apply_training_augmentation,
    random_crop,
)


VALID_MODES = ("random_crop", "full", "resize")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


class LEVIRCDDataset(Dataset):
    """
    PyTorch Dataset for the LEVIR-CD dataset.
    """

    def __init__(
        self,
        image_a_dir,
        image_b_dir,
        label_dir,
        image_size=256,
        file_names=None,
        augment=False,
        mode=None,
        crops_per_image=1,
        use_canny=USE_CANNY,
        photometric=PHOTOMETRIC_AUGMENTATION,
        change_focus_prob=CHANGE_FOCUSED_CROP_PROB,
    ):
        """
        Parameters
        ----------
        image_a_dir, image_b_dir, label_dir : str or Path
            Before / After / ground-truth directories.

        image_size : int
            Crop size ("random_crop") or target size ("resize").
            Ignored in "full" mode.

        file_names : list, optional
            Restrict the dataset to these filenames.

        augment : bool
            Apply synchronised geometric + independent photometric
            augmentation (training only).

        mode : {"random_crop", "full", "resize"} or None
            None -> "random_crop" if augment else "full".

        crops_per_image : int
            Number of random crops per image per epoch ("random_crop").

        use_canny : bool
            Append the Canny edge channel (4-channel input).
            False -> RGB only (3 channels), used for the ablation study.
        """

        self.image_a_dir = Path(image_a_dir)
        self.image_b_dir = Path(image_b_dir)
        self.label_dir = Path(label_dir)

        self.image_size = int(image_size)
        self.augment = augment
        self.mode = mode or ("random_crop" if augment else "full")
        self.crops_per_image = max(1, int(crops_per_image)) if self.mode == "random_crop" else 1
        self.use_canny = use_canny
        self.photometric = photometric
        self.change_focus_prob = change_focus_prob if augment else 0.0

        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")

        if file_names is None:
            self.file_names = self._find_valid_files()
        else:
            self.file_names = sorted(file_names)

        if len(self.file_names) == 0:
            raise RuntimeError(
                "No valid LEVIR-CD samples were found.\n"
                f"A directory     : {self.image_a_dir}\n"
                f"B directory     : {self.image_b_dir}\n"
                f"Label directory : {self.label_dir}"
            )

    # =====================================================
    # Find valid samples
    # =====================================================

    def _find_valid_files(self):
        """Filenames that exist in the A, B and label folders."""

        if not self.image_a_dir.is_dir():
            return []

        filenames = []

        for path in sorted(self.image_a_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if (self.image_b_dir / path.name).exists() and (self.label_dir / path.name).exists():
                filenames.append(path.name)

        return filenames

    @property
    def in_channels(self):
        return 4 if self.use_canny else 3

    # =====================================================
    # Dataset length
    # =====================================================

    def __len__(self):
        return len(self.file_names) * self.crops_per_image

    # =====================================================
    # Load one sample
    # =====================================================

    def load_raw(self, index):
        """
        Load the un-processed uint8 RGB images and uint8 mask
        of image `index` (no crop, no normalisation).
        """

        filename = self.file_names[index]

        image_a = load_rgb_image(self.image_a_dir / filename)
        image_b = load_rgb_image(self.image_b_dir / filename)
        mask = load_mask(self.label_dir / filename)

        return image_a, image_b, mask, filename

    def __getitem__(self, index):
        """
        Returns
        -------
        dict
            {
                "before":   Tensor C x H x W (standardised),
                "after":    Tensor C x H x W (standardised),
                "mask":     Tensor 1 x H x W in {0, 1},
                "filename": str
            }
        """

        image_index = index // self.crops_per_image
        image_a, image_b, mask, filename = self.load_raw(image_index)

        # -------------------------------------------------
        # Spatial sampling
        # -------------------------------------------------

        if self.mode == "random_crop":
            image_a, image_b, mask = random_crop(
                image_a,
                image_b,
                mask,
                self.image_size,
                change_focus_prob=self.change_focus_prob,
            )

        elif self.mode == "resize":
            image_a = resize_image(image_a, self.image_size)
            image_b = resize_image(image_b, self.image_size)
            mask = resize_mask(mask, self.image_size)

        # "full": keep native resolution

        # -------------------------------------------------
        # Augmentation (training only)
        # -------------------------------------------------

        if self.augment:
            image_a, image_b, mask = apply_training_augmentation(
                image_a,
                image_b,
                mask,
                photometric=self.photometric,
            )

        # -------------------------------------------------
        # DSP + standardisation  ->  H x W x C float32
        # -------------------------------------------------

        image_a = prepare_model_input(image_a, use_canny=self.use_canny)
        image_b = prepare_model_input(image_b, use_canny=self.use_canny)
        mask = normalize_mask(mask)

        # -------------------------------------------------
        # H x W x C  ->  C x H x W tensors
        # -------------------------------------------------

        image_a = torch.from_numpy(np.ascontiguousarray(image_a.transpose(2, 0, 1)))
        image_b = torch.from_numpy(np.ascontiguousarray(image_b.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)

        return {
            "before": image_a,
            "after": image_b,
            "mask": mask,
            "filename": filename,
        }
