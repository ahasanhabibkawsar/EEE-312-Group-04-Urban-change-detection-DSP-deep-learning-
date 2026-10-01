"""
Dataset package for Urban Change Detection.

LEVIR-CD image pairs are loaded at native resolution, augmented
(training only) and passed through the Gaussian + Canny DSP pipeline
(preprocessing/dsp_processing.py) to produce a standardised
4-channel RGB+Edge input for the Siamese U-Net.
"""

from .levir_dataset import LEVIRCDDataset
from .create_datasets import (
    create_train_dataset,
    create_validation_dataset,
    create_test_dataset,
    create_all_datasets,
)
from .dataloaders import create_dataloaders

__all__ = [
    "LEVIRCDDataset",
    "create_train_dataset",
    "create_validation_dataset",
    "create_test_dataset",
    "create_all_datasets",
    "create_dataloaders",
]
