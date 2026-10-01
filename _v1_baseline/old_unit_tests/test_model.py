"""
Test the complete Siamese U-Net model.
"""

import torch

from models.siamese_unet import SiameseUNet


def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    elif torch.cuda.is_available():

        return torch.device("cuda")

    else:

        return torch.device("cpu")


def main():

    print("=" * 60)
    print("COMPLETE SIAMESE U-NET TEST")
    print("=" * 60)

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    device = get_device()

    print()
    print("Device:", device)

    # -------------------------------------------------
    # Create model
    # -------------------------------------------------

    model = SiameseUNet(
        in_channels=4,
        base_channels=32,
    )

    model = model.to(device)

    # -------------------------------------------------
    # Print model
    # -------------------------------------------------

    print()
    print("Model:")
    print(model)

    # -------------------------------------------------
    # Dummy Before and After images
    # -------------------------------------------------

    before = torch.randn(
        4,
        4,
        256,
        256,
        device=device,
    )

    after = torch.randn(
        4,
        4,
        256,
        256,
        device=device,
    )

    # -------------------------------------------------
    # Forward pass
    # -------------------------------------------------

    with torch.no_grad():

        logits = model(
            before,
            after,
        )

    # -------------------------------------------------
    # Output
    # -------------------------------------------------

    print()
    print("Model output:")
    print(
        "Logits:",
        logits.shape,
    )

    # -------------------------------------------------
    # Expected shape
    # -------------------------------------------------

    expected_shape = (
        4,
        1,
        256,
        256,
    )

    print()
    print(
        "Expected:",
        expected_shape,
    )

    # -------------------------------------------------
    # Verify
    # -------------------------------------------------

    if tuple(logits.shape) == expected_shape:

        print()
        print("Output shape: CORRECT ✅")

    else:

        print()
        print("Output shape: INCORRECT ❌")

        raise RuntimeError(
            "Complete model output shape is incorrect."
        )

    # -------------------------------------------------
    # Test probability conversion
    # -------------------------------------------------

    probabilities = torch.sigmoid(
        logits
    )

    print()
    print(
        "Probability shape:",
        probabilities.shape,
    )

    print(
        "Probability range:",
        probabilities.min().item(),
        "to",
        probabilities.max().item(),
    )

    # -------------------------------------------------
    # Test binary prediction
    # -------------------------------------------------

    prediction = (
        probabilities >= 0.5
    ).float()

    print()
    print(
        "Prediction shape:",
        prediction.shape,
    )

    print(
        "Prediction values:",
        torch.unique(prediction).tolist(),
    )

    # -------------------------------------------------
    # Parameter count
    # -------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print("Total parameters:", total_parameters)
    print(
        "Trainable parameters:",
        trainable_parameters,
    )

    # -------------------------------------------------
    # Finish
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("COMPLETE SIAMESE U-NET TEST: PASSED")
    print("=" * 60)


if __name__ == "__main__":

    main()