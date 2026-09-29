"""
Test the U-Net decoder independently.
"""

import torch

from models.encoder import (
    SiameseEncoder,
)

from models.fusion import (
    SiameseFeatureFusion,
)

from models.decoder import (
    ChangeDecoder,
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
    print("U-NET DECODER TEST")
    print("=" * 60)

    # -------------------------------------------------
    # Device
    # -------------------------------------------------

    device = get_device()

    print()
    print("Device:", device)

    # -------------------------------------------------
    # Create encoder
    # -------------------------------------------------

    encoder = SiameseEncoder(
        in_channels=4,
        base_channels=32,
    )

    encoder = encoder.to(device)

    # -------------------------------------------------
    # Create fusion
    # -------------------------------------------------

    fusion = SiameseFeatureFusion(
        base_channels=32,
    )

    fusion = fusion.to(device)

    # -------------------------------------------------
    # Create decoder
    # -------------------------------------------------

    decoder = ChangeDecoder(
        base_channels=32,
    )

    decoder = decoder.to(device)

    # -------------------------------------------------
    # Dummy input
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
    # Encoder
    # -------------------------------------------------

    with torch.no_grad():

        encoder_output = encoder(
            before,
            after,
        )

    # -------------------------------------------------
    # Fusion
    # -------------------------------------------------

    with torch.no_grad():

        fusion_output = fusion(
            encoder_output
        )

    # -------------------------------------------------
    # Decoder
    # -------------------------------------------------

    with torch.no_grad():

        logits = decoder(
            fusion_output[
                "fused_features"
            ],

            fusion_output[
                "fused_bottleneck"
            ],
        )

    # -------------------------------------------------
    # Output shape
    # -------------------------------------------------

    print()
    print("Decoder output:")
    print(
        "Logits:",
        logits.shape,
    )

    print()
    print(
        "Expected:",
        torch.Size(
            [4, 1, 256, 256]
        ),
    )

    # -------------------------------------------------
    # Check output
    # -------------------------------------------------

    expected_shape = (
        4,
        1,
        256,
        256,
    )

    if tuple(logits.shape) == expected_shape:

        print()
        print("Output shape: CORRECT ✅")

    else:

        print()
        print("Output shape: INCORRECT ❌")

        raise RuntimeError(
            "Decoder output shape is incorrect."
        )

    # -------------------------------------------------
    # Finish
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("U-NET DECODER TEST: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()