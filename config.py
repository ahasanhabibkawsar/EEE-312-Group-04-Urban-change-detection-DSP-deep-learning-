"""
Central configuration for the Urban Change Detection project (v2).

Every script imports its settings from here, so a value changed in this
file changes the whole pipeline (training, evaluation, GUI, analyses).
"""

import json
from pathlib import Path


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------
# LEVIR-CD dataset
# ---------------------------------------------------------

DATA_ROOT = PROJECT_ROOT / "data" / "LEVIR-CD"

TRAIN_DIR = DATA_ROOT / "train"
TRAIN_IMAGE_A_DIR = TRAIN_DIR / "A"
TRAIN_IMAGE_B_DIR = TRAIN_DIR / "B"
TRAIN_LABEL_DIR = TRAIN_DIR / "label"

VAL_DIR = DATA_ROOT / "val"
VAL_IMAGE_A_DIR = VAL_DIR / "A"
VAL_IMAGE_B_DIR = VAL_DIR / "B"
VAL_LABEL_DIR = VAL_DIR / "label"

TEST_DIR = DATA_ROOT / "test"
TEST_IMAGE_A_DIR = TEST_DIR / "A"
TEST_IMAGE_B_DIR = TEST_DIR / "B"
TEST_LABEL_DIR = TEST_DIR / "label"


# ---------------------------------------------------------
# Image / patch configuration
# ---------------------------------------------------------
#
# LEVIR-CD images are 1024 x 1024 at 0.5 m/pixel.
#
# v1 resized them to 256 x 256 (4x down-sampling), so a 16 x 16 px
# building became 4 x 4 px. v2 keeps the native resolution:
#
#   training   -> random 256 x 256 crops of the 1024 x 1024 images
#   evaluation -> the full 1024 x 1024 image, predicted with an
#                 overlapping sliding window of 256 x 256 tiles
# ---------------------------------------------------------

IMAGE_SIZE = 256          # crop / tile size seen by the network
PATCH_SIZE = IMAGE_SIZE   # alias

TRAIN_CROPS_PER_IMAGE = 4        # random crops drawn per image per epoch
CHANGE_FOCUSED_CROP_PROB = 0.3   # chance a crop is centred on a changed pixel

INFERENCE_TILE_OVERLAP = 64      # sliding-window overlap (pixels)
INFERENCE_TILE_BATCH = 8         # tiles per forward pass during inference


# ---------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------
#
# RGB   : ImageNet statistics (the encoder is ImageNet pre-trained)
# Canny : binary map standardised with its own statistics.
#         Measured on 120 LEVIR-CD training images:
#         edge density p = 0.121  ->  std = sqrt(p(1-p)) = 0.326
#
# After normalisation all four channels have ~zero mean and unit
# variance, so the edge channel carries the same weight as R, G, B.
# ---------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EDGE_MEAN = 0.12
EDGE_STD = 0.33


# ---------------------------------------------------------
# DSP configuration
# ---------------------------------------------------------

USE_CANNY = True               # False = RGB-only ablation (3 channels)

GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA = 1.0

CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

MODEL_INPUT_CHANNELS = 4 if USE_CANNY else 3


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

BATCH_SIZE = 4
NUM_WORKERS = 2

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
WARMUP_EPOCHS = 1
EARLY_STOPPING_PATIENCE = 15
GRAD_CLIP_NORM = 1.0

# Multi-task loss weights
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
EDGE_LOSS_WEIGHT = 0.4

# Photometric augmentation (applied independently to each date to
# imitate season / illumination / sensor differences)
PHOTOMETRIC_AUGMENTATION = True

RANDOM_SEED = 42


# ---------------------------------------------------------
# Checkpoints and decision threshold
# ---------------------------------------------------------

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"
THRESHOLD_FILE = CHECKPOINT_DIR / "best_threshold.json"

DEFAULT_THRESHOLD = 0.5


def load_best_threshold(path=THRESHOLD_FILE, default=DEFAULT_THRESHOLD):
    """
    Return the decision threshold chosen on the VALIDATION set.

    train.py / evaluate.py --tune write it to checkpoints/best_threshold.json.
    If that file does not exist yet, DEFAULT_THRESHOLD is returned.
    """

    try:
        with open(path) as file:
            return float(json.load(file)["threshold"])
    except (OSError, KeyError, ValueError, TypeError):
        return default


def save_best_threshold(threshold, f1=None, path=THRESHOLD_FILE, source="validation"):
    """Write the validation-selected threshold to disk."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        json.dump(
            {"threshold": float(threshold), "val_f1": f1, "selected_on": source},
            file,
            indent=4,
        )
