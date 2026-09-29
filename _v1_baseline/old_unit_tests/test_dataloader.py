"""
Test the LEVIR-CD DataLoader pipeline.
"""

from dataset.dataloaders import (
    create_dataloaders,
)


def main():

    print("=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders()

    print()
    print("Train batches :", len(train_loader))
    print("Val batches   :", len(val_loader))
    print("Test batches  :", len(test_loader))

    # -------------------------------------------------
    # Get one training batch
    # -------------------------------------------------

    before, after, mask = None, None, None

    batch = next(iter(train_loader))

    before = batch["before"]
    after = batch["after"]
    mask = batch["mask"]

    print()
    print("Training batch")
    print("-" * 60)

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

    print(
        "Before dtype:",
        before.dtype
    )

    print(
        "After dtype :",
        after.dtype
    )

    print(
        "Mask dtype  :",
        mask.dtype
    )

    print(
        "Mask values:",
        mask.unique().tolist()
    )

    print()
    print("=" * 60)
    print("DATALOADER TEST: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()