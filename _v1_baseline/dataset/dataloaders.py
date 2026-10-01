"""
PyTorch DataLoader creation for LEVIR-CD.
"""

from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    NUM_WORKERS,
)

from dataset.create_datasets import (
    create_all_datasets,
)


def create_dataloaders():

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_all_datasets()

    train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    )

    val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    )

    test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )