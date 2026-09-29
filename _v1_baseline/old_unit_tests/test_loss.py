"""
Test the loss functions.
"""

import torch

from losses.losses import (
    DiceLoss,
    BCEDiceLoss,
)


def main():

    print("=" * 60)
    print("LOSS FUNCTION TEST")
    print("=" * 60)

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    if torch.backends.mps.is_available():

        device = torch.device("mps")

    elif torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print()
    print("Device:", device)

    # -------------------------------------------------
    # Dummy logits
    # -------------------------------------------------

    logits = torch.randn(
        4,
        1,
        256,
        256,
        device=device,
    )

    # -------------------------------------------------
    # Dummy ground truth
    # -------------------------------------------------

    targets = torch.randint(
        0,
        2,
        (
            4,
            1,
            256,
            256,
        ),
        device=device,
    ).float()

    # -------------------------------------------------
    # Dice loss
    # -------------------------------------------------

    dice_loss = DiceLoss()

    dice_value = dice_loss(
        logits,
        targets,
    )

    print()
    print(
        "Dice Loss:",
        dice_value.item(),
    )

    # -------------------------------------------------
    # BCE + Dice
    # -------------------------------------------------

    combined_loss = BCEDiceLoss()

    total_value = combined_loss(
        logits,
        targets,
    )

    print(
        "BCE + Dice Loss:",
        total_value.item(),
    )

    # -------------------------------------------------
    # Verify scalar
    # -------------------------------------------------

    if total_value.ndim == 0:

        print()
        print("Loss output: CORRECT ✅")

    else:

        raise RuntimeError(
            "Loss output is not scalar."
        )

    # -------------------------------------------------
    # Test backward
    # -------------------------------------------------

    logits.requires_grad_(True)

    loss = combined_loss(
        logits,
        targets,
    )

    loss.backward()

    if logits.grad is not None:

        print(
            "Backward propagation: CORRECT ✅"
        )

    else:

        raise RuntimeError(
            "Gradient was not calculated."
        )

    # -------------------------------------------------
    # Finish
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("LOSS FUNCTION TEST: PASSED")
    print("=" * 60)


if __name__ == "__main__":

    main()