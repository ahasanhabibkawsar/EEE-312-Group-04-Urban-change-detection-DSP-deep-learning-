"""
LEVIR-CD Dataset

Loads paired satellite images:

    A      -> image before change
    B      -> image after change
    label  -> ground-truth change mask

Processing pipeline:

    RGB image
        ↓
    Resize
        ↓
    Synchronized augmentation
        ↓
    DSP processing
        ↓
    RGB + Canny edge
        ↓
    4-channel tensor
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.image_processing import (
    load_rgb_image,
    load_mask,
    resize_image,
    resize_mask,
    normalize_mask,
)

from preprocessing.dsp_processing import (
    extract_dsp_features,
)

from dataset.augmentation import (
    apply_training_augmentation,
)


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
    ):
        """
        Parameters
        ----------
        image_a_dir : str or Path
            Directory containing Before images.

        image_b_dir : str or Path
            Directory containing After images.

        label_dir : str or Path
            Directory containing ground-truth masks.

        image_size : int
            Target image size.

        file_names : list, optional
            Specific filenames to use.

        augment : bool
            Whether to apply synchronized training augmentation.
        """

        self.image_a_dir = Path(image_a_dir)
        self.image_b_dir = Path(image_b_dir)
        self.label_dir = Path(label_dir)

        self.image_size = image_size
        self.augment = augment

        # -------------------------------------------------
        # Find valid filenames
        # -------------------------------------------------

        if file_names is None:

            self.file_names = self._find_valid_files()

        else:

            self.file_names = sorted(
                file_names
            )

        # -------------------------------------------------
        # Check dataset
        # -------------------------------------------------

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
        """
        Find filenames that exist in A, B and label folders.
        """

        extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
        )

        filenames = []

        for path in sorted(
            self.image_a_dir.iterdir()
        ):

            if not path.is_file():
                continue

            if path.suffix.lower() not in extensions:
                continue

            filename = path.name

            image_b_path = (
                self.image_b_dir / filename
            )

            label_path = (
                self.label_dir / filename
            )

            if (
                image_b_path.exists()
                and label_path.exists()
            ):

                filenames.append(
                    filename
                )

        return filenames

    # =====================================================
    # Dataset length
    # =====================================================

    def __len__(self):
        """
        Return number of samples.
        """

        return len(
            self.file_names
        )

    # =====================================================
    # Load one sample
    # =====================================================

    def __getitem__(self, index):
        """
        Load and process one sample.

        Returns
        -------
        dict
            {
                "before": Tensor,
                "after": Tensor,
                "mask": Tensor,
                "filename": str
            }
        """

        filename = self.file_names[index]

        # -------------------------------------------------
        # Construct file paths
        # -------------------------------------------------

        image_a_path = (
            self.image_a_dir / filename
        )

        image_b_path = (
            self.image_b_dir / filename
        )

        label_path = (
            self.label_dir / filename
        )

        # -------------------------------------------------
        # Load images
        # -------------------------------------------------

        image_a = load_rgb_image(
            image_a_path
        )

        image_b = load_rgb_image(
            image_b_path
        )

        mask = load_mask(
            label_path
        )

        # -------------------------------------------------
        # Resize
        # -------------------------------------------------

        image_a = resize_image(
            image_a,
            self.image_size
        )

        image_b = resize_image(
            image_b,
            self.image_size
        )

        mask = resize_mask(
            mask,
            self.image_size
        )

        # -------------------------------------------------
        # Synchronized augmentation
        # -------------------------------------------------
        #
        # IMPORTANT:
        #
        # The exact same spatial transformation is applied
        # to Before, After and Ground Truth.
        #
        # This is only enabled for training.
        # -------------------------------------------------

        if self.augment:

            (
                image_a,
                image_b,
                mask,
            ) = apply_training_augmentation(
                image_a,
                image_b,
                mask,
            )

        # -------------------------------------------------
        # DSP processing
        # -------------------------------------------------
        #
        # Each RGB image is converted into:
        #
        #     R
        #     G
        #     B
        #     Canny Edge
        #
        # Result:
        #
        #     H × W × 4
        # -------------------------------------------------

        image_a, _ = extract_dsp_features(
            image_a
        )

        image_b, _ = extract_dsp_features(
            image_b
        )

        # -------------------------------------------------
        # Normalize mask
        # -------------------------------------------------

        mask = normalize_mask(
            mask
        )

        # -------------------------------------------------
        # Convert H × W × C
        # to
        # C × H × W
        # -------------------------------------------------

        image_a = np.transpose(
            image_a,
            (2, 0, 1)
        )

        image_b = np.transpose(
            image_b,
            (2, 0, 1)
        )

        # -------------------------------------------------
        # Convert to PyTorch tensors
        # -------------------------------------------------

        image_a = torch.from_numpy(
            image_a.copy()
        ).float()

        image_b = torch.from_numpy(
            image_b.copy()
        ).float()

        mask = torch.from_numpy(
            mask.copy()
        ).float()

        # -------------------------------------------------
        # Add channel dimension to mask
        #
        # H × W
        #     ↓
        # 1 × H × W
        # -------------------------------------------------

        if mask.ndim == 2:

            mask = mask.unsqueeze(0)

        # -------------------------------------------------
        # Return sample
        # -------------------------------------------------

        return {
            "before": image_a,
            "after": image_b,
            "mask": mask,
            "filename": filename,
        }