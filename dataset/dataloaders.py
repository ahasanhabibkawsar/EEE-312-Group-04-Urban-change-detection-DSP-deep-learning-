"""
PyTorch DataLoader creation for LEVIR-CD.

Validation / test loaders use batch_size=1 because every sample is a
full 1024 x 1024 image; the model tiles it internally
(see SiameseUNet.sliding_window_forward).
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, NUM_WORKERS, USE_CANNY, TRAIN_CROPS_PER_IMAGE
from dataset.create_datasets import create_all_datasets


def seed_worker(worker_id):
    """Give every DataLoader worker its own reproducible NumPy / random seed."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    use_canny=USE_CANNY,
    crops_per_image=TRAIN_CROPS_PER_IMAGE,
    seed=None,
):
    train_dataset, val_dataset, test_dataset = create_all_datasets(
        use_canny=use_canny,
        crops_per_image=crops_per_image,
    )

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    common = dict(
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,          # avoids a size-1 batch in BatchNorm
        generator=generator,
        **common,
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **common)

    return train_loader, val_loader, test_loader
