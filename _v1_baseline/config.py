from pathlib import Path


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------
# LEVIR-CD dataset
# ---------------------------------------------------------

DATA_ROOT = PROJECT_ROOT / "data" / "LEVIR-CD"


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

TRAIN_DIR = DATA_ROOT / "train"

TRAIN_IMAGE_A_DIR = TRAIN_DIR / "A"
TRAIN_IMAGE_B_DIR = TRAIN_DIR / "B"
TRAIN_LABEL_DIR = TRAIN_DIR / "label"


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

VAL_DIR = DATA_ROOT / "val"

VAL_IMAGE_A_DIR = VAL_DIR / "A"
VAL_IMAGE_B_DIR = VAL_DIR / "B"
VAL_LABEL_DIR = VAL_DIR / "label"


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

TEST_DIR = DATA_ROOT / "test"

TEST_IMAGE_A_DIR = TEST_DIR / "A"
TEST_IMAGE_B_DIR = TEST_DIR / "B"
TEST_LABEL_DIR = TEST_DIR / "label"


# ---------------------------------------------------------
# Image configuration
# ---------------------------------------------------------

IMAGE_SIZE = 256

MODEL_INPUT_CHANNELS = 4


# ---------------------------------------------------------
# DSP configuration
# ---------------------------------------------------------

GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA = 1.0

CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

BATCH_SIZE = 4
NUM_WORKERS = 2

LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

RANDOM_SEED = 42