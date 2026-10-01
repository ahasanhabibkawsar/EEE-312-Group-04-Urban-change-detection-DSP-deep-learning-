"""
Main entry point for the Urban Change Detection project.

Current stage:
    1. Load LEVIR-CD datasets
    2. Verify train/validation/test split
    3. Verify preprocessing
"""

from dataset.create_datasets import (
    create_all_datasets,
)


def print_dataset_information(
    train_dataset,
    val_dataset,
    test_dataset,
):
    """
    Display dataset information.
    """

    print()
    print("=" * 60)
    print("       URBAN CHANGE DETECTION PROJECT")
    print("=" * 60)

    print()
    print("Dataset: LEVIR-CD")
    print()

    print(
        f"Training samples   : {len(train_dataset)}"
    )

    print(
        f"Validation samples : {len(val_dataset)}"
    )

    print(
        f"Testing samples    : {len(test_dataset)}"
    )

    print(
        f"Total samples      : "
        f"{len(train_dataset) + len(val_dataset) + len(test_dataset)}"
    )

    print()
    print("=" * 60)


def test_dataset_sample(
    train_dataset,
):
    """
    Test one training sample and display its shapes.
    """

    sample = train_dataset[0]

    print()
    print("Sample verification")
    print("-" * 60)

    print(
        f"Filename       : {sample['filename']}"
    )

    print(
        f"Before shape   : {sample['before'].shape}"
    )

    print(
        f"After shape    : {sample['after'].shape}"
    )

    print(
        f"Ground truth   : {sample['mask'].shape}"
    )

    print(
        f"Before range   : "
        f"{sample['before'].min().item():.2f} "
        f"to "
        f"{sample['before'].max().item():.2f}"
    )

    print(
        f"After range    : "
        f"{sample['after'].min().item():.2f} "
        f"to "
        f"{sample['after'].max().item():.2f}"
    )

    print(
        f"Mask values    : "
        f"{sample['mask'].unique().tolist()}"
    )

    print("-" * 60)


def main():
    """
    Main project controller.
    """

    print()
    print("Starting Urban Change Detection Project...")

    # -------------------------------------------------
    # Load datasets
    # -------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_all_datasets()

    # -------------------------------------------------
    # Display dataset information
    # -------------------------------------------------

    print_dataset_information(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    # -------------------------------------------------
    # Test preprocessing pipeline
    # -------------------------------------------------

    test_dataset_sample(
        train_dataset
    )

    print()
    print("Dataset and preprocessing pipeline: OK")
    print()


if __name__ == "__main__":
    main()