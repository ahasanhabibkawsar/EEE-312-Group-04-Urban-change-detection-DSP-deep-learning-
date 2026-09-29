"""
Training Pipeline Integration Test

Checks:

Dataset
    ↓
DataLoader
    ↓
Siamese U-Net
    ↓
Loss
    ↓
Backpropagation
"""

import torch

from dataset import create_dataloaders
from models import SiameseUNet
from losses import BCEDiceLoss


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    elif torch.cuda.is_available():

        return torch.device("cuda")

    else:

        return torch.device("cpu")


# ============================================================
# CHECK BATCH
# ============================================================

def inspect_batch(batch):

    print()
    print("Batch type:")
    print(type(batch))

    if not isinstance(batch, dict):

        raise TypeError(
            "Expected DataLoader batch to be a dictionary."
        )

    print()
    print("Batch keys:")

    for key, value in batch.items():

        if torch.is_tensor(value):

            print(
                f"{key}: {value.shape}"
            )

        elif isinstance(value, list):

            print(
                f"{key}: "
                f"list (length={len(value)})"
            )

        else:

            print(
                f"{key}: "
                f"{type(value).__name__}"
            )

    # --------------------------------------------------------
    # Required keys
    # --------------------------------------------------------

    required_keys = [
        "before",
        "after",
        "mask",
    ]

    for key in required_keys:

        if key not in batch:

            raise KeyError(
                f"Missing required key: {key}"
            )

    print()
    print("Required keys: CORRECT ✅")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TRAINING PIPELINE INTEGRATION TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print()
    print("Device:", device)

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    print()
    print("Creating DataLoaders...")

    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    print()
    print("Train batches :", len(train_loader))
    print("Val batches   :", len(val_loader))
    print("Test batches  :", len(test_loader))

    # --------------------------------------------------------
    # Get one batch
    # --------------------------------------------------------

    batch = next(
        iter(train_loader)
    )

    # --------------------------------------------------------
    # Inspect batch
    # --------------------------------------------------------

    inspect_batch(batch)

    # --------------------------------------------------------
    # Move tensors to device
    # --------------------------------------------------------

    before = batch["before"].to(
        device
    )

    after = batch["after"].to(
        device
    )

    mask = batch["mask"].to(
        device
    )

    print()
    print("Training tensors:")

    print(
        "Before:",
        before.shape
    )

    print(
        "After :",
        after.shape
    )

    print(
        "Mask  :",
        mask.shape
    )

    # --------------------------------------------------------
    # Check dtype
    # --------------------------------------------------------

    print()
    print("Data types:")

    print(
        "Before:",
        before.dtype
    )

    print(
        "After :",
        after.dtype
    )

    print(
        "Mask  :",
        mask.dtype
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print("Creating Siamese U-Net...")

    model = SiameseUNet(
        in_channels=4,
        base_channels=32,
    )

    model = model.to(device)

    print(
        "Model created: CORRECT ✅"
    )

    # --------------------------------------------------------
    # Create loss
    # --------------------------------------------------------

    criterion = BCEDiceLoss()

    print(
        "Loss function created: CORRECT ✅"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    print()
    print("Running forward pass...")

    logits = model(
        before,
        after,
    )

    print(
        "Logits:",
        logits.shape
    )

    # --------------------------------------------------------
    # Verify output shape
    # --------------------------------------------------------

    if logits.shape != mask.shape:

        raise RuntimeError(
            "Model output shape does not "
            "match target mask shape."
        )

    print(
        "Output shape: CORRECT ✅"
    )

    # --------------------------------------------------------
    # Calculate loss
    # --------------------------------------------------------

    loss = criterion(
        logits,
        mask,
    )

    print()
    print(
        "Initial loss:",
        loss.item()
    )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    print()
    print("Testing backward propagation...")

    model.zero_grad()

    loss.backward()

    print(
        "Backward propagation: CORRECT ✅"
    )

    # --------------------------------------------------------
    # Check gradients
    # --------------------------------------------------------

    gradient_count = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_count += 1

    print()
    print(
        "Parameters with gradients:",
        gradient_count
    )

    if gradient_count == 0:

        raise RuntimeError(
            "No gradients were generated."
        )

    print(
        "Gradient flow: CORRECT ✅"
    )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING PIPELINE TEST: PASSED ✅")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()