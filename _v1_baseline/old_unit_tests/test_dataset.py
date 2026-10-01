"""
Test the LEVIR-CD dataset and DSP preprocessing.
"""

import matplotlib.pyplot as plt
import numpy as np

from config import (
    IMAGE_A_DIR,
    IMAGE_B_DIR,
    LABEL_DIR,
    IMAGE_SIZE,
)

from dataset.levir_dataset import (
    LEVIRCDDataset,
)


def display_sample(sample):

    before = sample["before"].numpy()
    after = sample["after"].numpy()
    mask = sample["mask"].numpy()

    # CHW -> HWC
    before_rgb = np.transpose(
        before[:3],
        (1, 2, 0)
    )

    after_rgb = np.transpose(
        after[:3],
        (1, 2, 0)
    )

    before_edge = before[3]

    after_edge = after[3]

    mask = mask[0]

    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(before_rgb)
    plt.title("Before Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(after_rgb)
    plt.title("After Image")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(before_edge, cmap="gray")
    plt.title("Before - Canny Edge")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(after_edge, cmap="gray")
    plt.title("After - Canny Edge")
    plt.axis("off")

    plt.tight_layout()

    plt.show()


def main():

    dataset = LEVIRCDDataset(
        image_a_dir=IMAGE_A_DIR,
        image_b_dir=IMAGE_B_DIR,
        label_dir=LABEL_DIR,
        image_size=IMAGE_SIZE,
    )

    print("=" * 50)
    print("LEVIR-CD DATASET TEST")
    print("=" * 50)

    print("Number of samples:", len(dataset))

    sample = dataset[0]

    print("\nSample information:")
    print("Filename :", sample["filename"])

    print(
        "Before shape:",
        sample["before"].shape
    )

    print(
        "After shape :",
        sample["after"].shape
    )

    print(
        "Mask shape  :",
        sample["mask"].shape
    )

    print(
        "\nBefore range:",
        sample["before"].min().item(),
        "to",
        sample["before"].max().item()
    )

    print(
        "After range:",
        sample["after"].min().item(),
        "to",
        sample["after"].max().item()
    )

    print(
        "Mask values:",
        sample["mask"].unique().tolist()
    )

    display_sample(sample)


if __name__ == "__main__":
    main()