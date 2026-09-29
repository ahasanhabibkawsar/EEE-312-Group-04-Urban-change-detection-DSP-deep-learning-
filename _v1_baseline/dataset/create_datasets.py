"""
Create the official LEVIR-CD train,
validation and test datasets.
"""

from config import (
    TRAIN_IMAGE_A_DIR,
    TRAIN_IMAGE_B_DIR,
    TRAIN_LABEL_DIR,

    VAL_IMAGE_A_DIR,
    VAL_IMAGE_B_DIR,
    VAL_LABEL_DIR,

    TEST_IMAGE_A_DIR,
    TEST_IMAGE_B_DIR,
    TEST_LABEL_DIR,

    IMAGE_SIZE,
)

from dataset.levir_dataset import (
    LEVIRCDDataset,
)


def create_train_dataset():

    return LEVIRCDDataset(
        image_a_dir=TRAIN_IMAGE_A_DIR,
        image_b_dir=TRAIN_IMAGE_B_DIR,
        label_dir=TRAIN_LABEL_DIR,
        image_size=IMAGE_SIZE,
        augment=True,
    )


def create_validation_dataset():

    return LEVIRCDDataset(
        image_a_dir=VAL_IMAGE_A_DIR,
        image_b_dir=VAL_IMAGE_B_DIR,
        label_dir=VAL_LABEL_DIR,
        image_size=IMAGE_SIZE,
        augment=False,
    )


def create_test_dataset():

    return LEVIRCDDataset(
        image_a_dir=TEST_IMAGE_A_DIR,
        image_b_dir=TEST_IMAGE_B_DIR,
        label_dir=TEST_LABEL_DIR,
        image_size=IMAGE_SIZE,
        augment=False,
    )


def create_all_datasets():

    train_dataset = create_train_dataset()

    val_dataset = create_validation_dataset()

    test_dataset = create_test_dataset()

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


if __name__ == "__main__":

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_all_datasets()

    print("=" * 50)
    print("LEVIR-CD DATASET")
    print("=" * 50)

    print(
        "Training samples   :",
        len(train_dataset)
    )

    print(
        "Validation samples :",
        len(val_dataset)
    )

    print(
        "Testing samples    :",
        len(test_dataset)
    )

    print(
        "Total samples      :",
        len(train_dataset)
        + len(val_dataset)
        + len(test_dataset)
    )

    print("=" * 50)