"""
Test the Siamese Encoder.
"""

import torch

from models.encoder import SiameseEncoder


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    elif torch.cuda.is_available():
        return torch.device("cuda")

    else:
        return torch.device("cpu")


def main():

    print("=" * 60)
    print("SIAMESE ENCODER TEST")
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

    model = SiameseEncoder(
        in_channels=4,
        base_channels=32,
    )

    model = model.to(device)

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

        output = model(
            before,
            after,
        )

    # -------------------------------------------------
    # Before features
    # -------------------------------------------------

    print()
    print("Before features:")
    print("-" * 40)

    for index, feature in enumerate(
        output["before_features"],
        start=1,
    ):

        print(
            f"Stage {index}: {feature.shape}"
        )

    print()
    print(
        "Before bottleneck:",
        output["before_bottleneck"].shape,
    )

    # -------------------------------------------------
    # After features
    # -------------------------------------------------

    print()
    print("After features:")
    print("-" * 40)

    for index, feature in enumerate(
        output["after_features"],
        start=1,
    ):

        print(
            f"Stage {index}: {feature.shape}"
        )

    print()
    print(
        "After bottleneck:",
        output["after_bottleneck"].shape,
    )

    # -------------------------------------------------
    # Finish
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("SIAMESE ENCODER TEST: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()