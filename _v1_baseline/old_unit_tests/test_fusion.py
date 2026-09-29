"""
Test the feature fusion module.
"""

import torch

from models.encoder import (
    SiameseEncoder,
)

from models.fusion import (
    SiameseFeatureFusion,
)


def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    elif torch.cuda.is_available():

        return torch.device("cuda")

    else:

        return torch.device("cpu")


def main():

    print("=" * 60)
    print("FEATURE FUSION TEST")
    print("=" * 60)

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    device = get_device()

    print()
    print("Device:", device)

    # -------------------------------------------------
    # Encoder
    # -------------------------------------------------

    encoder = SiameseEncoder(
        in_channels=4,
        base_channels=32,
    )

    encoder = encoder.to(device)

    # -------------------------------------------------
    # Fusion module
    # -------------------------------------------------

    fusion = SiameseFeatureFusion(
        base_channels=32,
    )

    fusion = fusion.to(device)

    # -------------------------------------------------
    # Dummy inputs
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
    # Encoder forward
    # -------------------------------------------------

    with torch.no_grad():

        encoder_output = encoder(
            before,
            after,
        )

    # -------------------------------------------------
    # Fusion forward
    # -------------------------------------------------

    with torch.no_grad():

        fusion_output = fusion(
            encoder_output
        )

    # -------------------------------------------------
    # Print fused features
    # -------------------------------------------------

    print()
    print("Fused encoder features:")
    print("-" * 40)

    for index, feature in enumerate(
        fusion_output[
            "fused_features"
        ],
        start=1,
    ):

        print(
            f"Stage {index}:",
            feature.shape,
        )

    # -------------------------------------------------
    # Bottleneck
    # -------------------------------------------------

    print()
    print(
        "Fused bottleneck:",
        fusion_output[
            "fused_bottleneck"
        ].shape,
    )

    # -------------------------------------------------
    # Finish
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("FEATURE FUSION TEST: PASSED")
    print("=" * 60)


if __name__ == "__main__":

    main()