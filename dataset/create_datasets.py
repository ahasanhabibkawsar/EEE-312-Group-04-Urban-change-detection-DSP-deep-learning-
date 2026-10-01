"""
Create the official LEVIR-CD train, validation and test datasets.

train -> native-resolution random crops + augmentation
val   -> full 1024 x 1024 images (sliding-window inference)
test  -> full 1024 x 1024 images (sliding-window inference)
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
    TRAIN_CROPS_PER_IMAGE,
    USE_CANNY,
)

from dataset.levir_dataset import LEVIRCDDataset


def create_train_dataset(use_canny=USE_CANNY, crops_per_image=TRAIN_CROPS_PER_IMAGE):
    return LEVIRCDDataset(
        image_a_dir=TRAIN_IMAGE_A_DIR,
        image_b_dir=TRAIN_IMAGE_B_DIR,
        label_dir=TRAIN_LABEL_DIR,
        image_size=IMAGE_SIZE,
        augment=True,
        mode="random_crop",
        crops_per_image=crops_per_image,
        use_canny=use_canny,
    )


def create_validation_dataset(use_canny=USE_CANNY):
    return LEVIRCDDataset(
        image_a_dir=VAL_IMAGE_A_DIR,
        image_b_dir=VAL_IMAGE_B_DIR,
        label_dir=VAL_LABEL_DIR,
        image_size=IMAGE_SIZE,
        augment=False,
        mode="full",
        use_canny=use_canny,
    )


def create_test_dataset(use_canny=USE_CANNY):
    return LEVIRCDDataset(
        image_a_dir=TEST_IMAGE_A_DIR,
        image_b_dir=TEST_IMAGE_B_DIR,
        label_dir=TEST_LABEL_DIR,
        image_size=IMAGE_SIZE,
        augment=False,
        mode="full",
        use_canny=use_canny,
    )


def create_all_datasets(use_canny=USE_CANNY, crops_per_image=TRAIN_CROPS_PER_IMAGE):
    return (
        create_train_dataset(use_canny, crops_per_image),
        create_validation_dataset(use_canny),
        create_test_dataset(use_canny),
    )


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = create_all_datasets()

    print("=" * 50)
    print("LEVIR-CD DATASET")
    print("=" * 50)
    print("Training images    :", len(train_dataset.file_names),
          f"({len(train_dataset)} random crops / epoch)")
    print("Validation images  :", len(val_dataset))
    print("Testing images     :", len(test_dataset))
    print("=" * 50)
