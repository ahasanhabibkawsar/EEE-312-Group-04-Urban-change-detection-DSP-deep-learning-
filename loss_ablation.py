# ================================================================
# URBAN CHANGE DETECTION
# LOSS FUNCTION ABLATION STUDY
#
# File:
# /Users/ahasanhabibkawsar/Documents/Urban_Change_Detection/loss_ablation.py
# ================================================================

import os
import sys
import json
import csv
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ================================================================
# PROJECT PATH
# ================================================================

PROJECT_ROOT = Path(
    "/Users/ahasanhabibkawsar/Documents/Urban_Change_Detection"
)

sys.path.insert(0, str(PROJECT_ROOT))

# ================================================================
# PROJECT IMPORTS
# ================================================================

from dataset.levir_dataset import LEVIRCDDataset
from models.siamese_unet import SiameseUNet
from config import load_best_threshold
from preprocessing.dsp_processing import denormalize_for_display


# ================================================================
# CONFIGURATION
# ================================================================

DATASET_ROOT = (
    PROJECT_ROOT / "data" / "LEVIR-CD"
)

TRAIN_A = DATASET_ROOT / "train" / "A"
TRAIN_B = DATASET_ROOT / "train" / "B"
TRAIN_LABEL = DATASET_ROOT / "train" / "label"

OUTPUT_DIR = (
    PROJECT_ROOT / "loss_ablation_results"
)

IMAGE_SIZE = 256
BATCH_SIZE = 4

EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

THRESHOLD = load_best_threshold()  # v2: threshold selected on the VALIDATION set

NUM_WORKERS = 0

SEED = 42

LOSS_FUNCTIONS = [
    "BCE",
    "Dice",
    "BCE + Dice",
    "Focal",
    "BCE + Focal",
]


# ================================================================
# RANDOM SEED
# ================================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================
# DEVICE
# ================================================================

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ================================================================
# DIRECTORY CHECK
# ================================================================

def check_directories():

    required = [
        DATASET_ROOT,
        TRAIN_A,
        TRAIN_B,
        TRAIN_LABEL,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required directory not found:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nAll required directories found. ✅")


# ================================================================
# FIND DATASET FILES
# ================================================================

def find_valid_filenames():

    extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff"
    }

    filenames = []

    for path in sorted(TRAIN_A.iterdir()):

        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        filename = path.name

        if not (TRAIN_B / filename).exists():
            continue

        if not (TRAIN_LABEL / filename).exists():
            continue

        filenames.append(filename)

    return sorted(filenames)


# ================================================================
# CREATE DATASETS
# ================================================================

def create_dataloaders():

    filenames = find_valid_filenames()

    print(
        f"\nTotal valid training samples: {len(filenames)}"
    )

    if len(filenames) < 2:

        raise RuntimeError(
            "Not enough training samples."
        )

    # ------------------------------------------------------------
    # Reproducible train/validation split
    # ------------------------------------------------------------

    rng = random.Random(SEED)

    shuffled = filenames.copy()

    rng.shuffle(shuffled)

    val_count = 64

    if len(shuffled) <= val_count:

        raise RuntimeError(
            "Dataset is too small for validation split."
        )

    val_files = sorted(
        shuffled[:val_count]
    )

    train_files = sorted(
        shuffled[val_count:]
    )

    print(
        f"Train samples : {len(train_files)}"
    )

    print(
        f"Val samples   : {len(val_files)}"
    )

    # ------------------------------------------------------------
    # Training dataset
    # ------------------------------------------------------------

    train_dataset = LEVIRCDDataset(

        image_a_dir=TRAIN_A,
        image_b_dir=TRAIN_B,
        label_dir=TRAIN_LABEL,

        image_size=IMAGE_SIZE,

        file_names=train_files,

        augment=True
    )

    # ------------------------------------------------------------
    # Validation dataset
    # ------------------------------------------------------------

    val_dataset = LEVIRCDDataset(

        image_a_dir=TRAIN_A,
        image_b_dir=TRAIN_B,
        label_dir=TRAIN_LABEL,

        image_size=IMAGE_SIZE,

        file_names=val_files,

        augment=False
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=False
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=False
    )

    print(
        f"Train batches : {len(train_loader)}"
    )

    print(
        f"Val batches   : {len(val_loader)}"
    )

    return train_loader, val_loader


# ================================================================
# DICE LOSS
# ================================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):

        super().__init__()

        self.smooth = smooth

    def forward(self, prediction, target):

        prediction = prediction.contiguous()
        target = target.contiguous()

        prediction = prediction.view(
            prediction.size(0),
            -1
        )

        target = target.view(
            target.size(0),
            -1
        )

        intersection = (
            prediction * target
        ).sum(dim=1)

        dice = (

            2.0 * intersection
            + self.smooth

        ) / (

            prediction.sum(dim=1)
            + target.sum(dim=1)
            + self.smooth

        )

        return (
            1.0 - dice
        ).mean()


# ================================================================
# FOCAL LOSS
# ================================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prediction, target):

        eps = 1e-7

        prediction = torch.clamp(
            prediction,
            eps,
            1.0 - eps
        )

        target = target.float()

        pt = (

            target * prediction

            +

            (1.0 - target)
            * (1.0 - prediction)

        )

        alpha_t = (

            target * self.alpha

            +

            (1.0 - target)
            * (1.0 - self.alpha)

        )

        focal_weight = (

            alpha_t
            * torch.pow(
                1.0 - pt,
                self.gamma
            )

        )

        bce = -(

            target
            * torch.log(prediction)

            +

            (1.0 - target)
            * torch.log(1.0 - prediction)

        )

        return (
            focal_weight * bce
        ).mean()


# ================================================================
# CREATE LOSS
# ================================================================

def create_loss(loss_name):

    bce = nn.BCELoss()

    dice = DiceLoss()

    focal = FocalLoss()

    if loss_name == "BCE":

        return bce

    if loss_name == "Dice":

        return dice

    if loss_name == "BCE + Dice":

        return lambda prediction, target: (

            bce(prediction, target)
            +
            dice(prediction, target)

        )

    if loss_name == "Focal":

        return focal

    if loss_name == "BCE + Focal":

        return lambda prediction, target: (

            bce(prediction, target)
            +
            focal(prediction, target)

        )

    raise ValueError(
        f"Unknown loss: {loss_name}"
    )


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(
    predictions,
    targets,
    threshold=0.45
):

    predictions = (
        predictions >= threshold
    ).float()

    targets = (
        targets >= 0.5
    ).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    tp = (
        (predictions == 1)
        &
        (targets == 1)
    ).sum().item()

    tn = (
        (predictions == 0)
        &
        (targets == 0)
    ).sum().item()

    fp = (
        (predictions == 1)
        &
        (targets == 0)
    ).sum().item()

    fn = (
        (predictions == 0)
        &
        (targets == 1)
    ).sum().item()

    epsilon = 1e-8

    accuracy = (
        (tp + tn)
        /
        (tp + tn + fp + fn + epsilon)
    )

    precision = (
        tp
        /
        (tp + fp + epsilon)
    )

    recall = (
        tp
        /
        (tp + fn + epsilon)
    )

    f1 = (

        2.0
        * precision
        * recall

        /

        (precision + recall + epsilon)

    )

    iou = (

        tp
        /

        (tp + fp + fn + epsilon)

    )

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "iou": iou,

        "tp": tp,

        "tn": tn,

        "fp": fp,

        "fn": fn,

    }


# ================================================================
# TRAIN ONE EPOCH
# ================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
    total_epochs
):

    model.train()

    running_loss = 0.0

    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader):

        # --------------------------------------------------------
        # IMPORTANT:
        # LEVIRCDDataset returns a dictionary
        # --------------------------------------------------------

        image_a = batch["before"].to(
            device,
            non_blocking=True
        )

        image_b = batch["after"].to(
            device,
            non_blocking=True
        )

        target = batch["mask"].to(
            device,
            non_blocking=True
        )

        # --------------------------------------------------------
        # Forward
        # --------------------------------------------------------

        prediction = model(
            image_a,
            image_b
        )

        # --------------------------------------------------------
        # Loss
        # --------------------------------------------------------

        loss = criterion(
            prediction,
            target
        )

        # --------------------------------------------------------
        # Backpropagation
        # --------------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
        )

        # --------------------------------------------------------
        # Progress
        # --------------------------------------------------------

        if (
            batch_idx == 0
            or
            (batch_idx + 1) % 25 == 0
            or
            (batch_idx + 1) == total_batches
        ):

            print(
                f"Epoch "
                f"{epoch}/{total_epochs} | "
                f"Batch "
                f"{batch_idx + 1}/{total_batches} | "
                f"Loss "
                f"{loss.item():.6f}"
            )

    return (
        running_loss
        /
        max(total_batches, 1)
    )


# ================================================================
# VALIDATION
# ================================================================

def validate(
    model,
    loader,
    criterion,
    device,
    threshold
):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for batch in loader:

            image_a = batch[
                "before"
            ].to(device)

            image_b = batch[
                "after"
            ].to(device)

            target = batch[
                "mask"
            ].to(device)

            prediction = model(
                image_a,
                image_b
            )

            loss = criterion(
                prediction,
                target
            )

            running_loss += (
                loss.item()
            )

            all_predictions.append(
                prediction.detach()
            )

            all_targets.append(
                target.detach()
            )

    predictions = torch.cat(
        all_predictions,
        dim=0
    )

    targets = torch.cat(
        all_targets,
        dim=0
    )

    metrics = calculate_metrics(

        predictions,

        targets,

        threshold

    )

    validation_loss = (

        running_loss
        /
        max(len(loader), 1)

    )

    metrics["loss"] = validation_loss

    return metrics


# ================================================================
# CREATE MODEL
# ================================================================

def create_model(device):

    # IMPORTANT:
    # Your actual SiameseUNet accepts in_channels.
    # Do NOT use out_channels here.

    model = SiameseUNet(
        in_channels=4
    )

    model = model.to(device)

    return model


# ================================================================
# COUNT PARAMETERS
# ================================================================

def print_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters    : {total:,}"
    )

    print(
        f"Trainable parameters: {trainable:,}"
    )


# ================================================================
# SAVE JSON
# ================================================================

def save_json(data, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


# ================================================================
# RUN ONE ABLATION EXPERIMENT
# ================================================================

def run_experiment(
    loss_name,
    train_loader,
    val_loader,
    device
):

    print("\n")
    print("=" * 68)

    print(
        f"LOSS ABLATION EXPERIMENT: "
        f"{loss_name}"
    )

    print("=" * 68)

    print(
        f"\nDevice       : {device}"
    )

    print(
        f"Loss         : {loss_name}"
    )

    print(
        f"Epochs       : {EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Weight decay : {WEIGHT_DECAY}"
    )

    print(
        f"Threshold    : {THRESHOLD}"
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = create_model(
        device
    )

    print_parameters(
        model
    )

    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------

    criterion = create_loss(
        loss_name
    )

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )

    # ------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=5

    )

    # ------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------

    safe_name = (
        loss_name
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )

    experiment_dir = (
        OUTPUT_DIR / safe_name
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_checkpoint = (
        experiment_dir
        /
        "best_model.pth"
    )

    # ------------------------------------------------------------
    # History
    # ------------------------------------------------------------

    history = []

    best_f1 = -1.0

    best_epoch = 0

    start_time = time.time()

    # ============================================================
    # TRAINING
    # ============================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print("\n")
        print(
            "-" * 68
        )

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Learning rate: "
            f"{current_lr:.8f}"
        )

        # --------------------------------------------------------
        # Training
        # --------------------------------------------------------

        train_loss = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            device,

            epoch,

            EPOCHS

        )

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        validation = validate(

            model,

            val_loader,

            criterion,

            device,

            THRESHOLD

        )

        val_loss = validation[
            "loss"
        ]

        val_f1 = validation[
            "f1"
        ]

        val_iou = validation[
            "iou"
        ]

        val_precision = validation[
            "precision"
        ]

        val_recall = validation[
            "recall"
        ]

        print(
            f"\nTrain Loss : "
            f"{train_loss:.6f}"
        )

        print(
            f"Val Loss   : "
            f"{val_loss:.6f}"
        )

        print(
            f"Precision  : "
            f"{val_precision:.6f}"
        )

        print(
            f"Recall     : "
            f"{val_recall:.6f}"
        )

        print(
            f"F1         : "
            f"{val_f1:.6f}"
        )

        print(
            f"IoU        : "
            f"{val_iou:.6f}"
        )

        # --------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------

        scheduler.step(
            val_f1
        )

        # --------------------------------------------------------
        # History
        # --------------------------------------------------------

        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "val_loss": val_loss,

            "precision": val_precision,

            "recall": val_recall,

            "f1": val_f1,

            "iou": val_iou,

            "learning_rate":
                optimizer.param_groups[0]["lr"]

        })

        # --------------------------------------------------------
        # Save best model
        # --------------------------------------------------------

        if val_f1 > best_f1:

            best_f1 = val_f1

            best_epoch = epoch

            torch.save(

                {

                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "best_f1": best_f1,

                    "loss_name":
                        loss_name,

                },

                best_checkpoint

            )

            print(
                "\n⭐ New best model saved!"
            )

    # ============================================================
    # FINAL BEST MODEL
    # ============================================================

    runtime = (
        time.time()
        -
        start_time
    )

    # Load best checkpoint

    checkpoint = torch.load(

        best_checkpoint,

        map_location=device,

        weights_only=False

    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # Final validation

    final_metrics = validate(

        model,

        val_loader,

        criterion,

        device,

        THRESHOLD

    )

    result = {

        "loss": loss_name,

        "best_epoch": best_epoch,

        "best_f1": final_metrics[
            "f1"
        ],

        "best_iou": final_metrics[
            "iou"
        ],

        "precision": final_metrics[
            "precision"
        ],

        "recall": final_metrics[
            "recall"
        ],

        "accuracy": final_metrics[
            "accuracy"
        ],

        "val_loss": final_metrics[
            "loss"
        ],

        "runtime_seconds": runtime,

        "checkpoint":
            str(best_checkpoint),

        "history": history

    }

    save_json(

        result,

        experiment_dir
        /
        "results.json"

    )

    print("\n")
    print(
        "=" * 68
    )

    print(
        f"EXPERIMENT COMPLETE: "
        f"{loss_name}"
    )

    print(
        "=" * 68
    )

    print(
        f"Best epoch : "
        f"{best_epoch}"
    )

    print(
        f"F1         : "
        f"{result['best_f1']:.6f}"
    )

    print(
        f"IoU        : "
        f"{result['best_iou']:.6f}"
    )

    print(
        f"Precision  : "
        f"{result['precision']:.6f}"
    )

    print(
        f"Recall     : "
        f"{result['recall']:.6f}"
    )

    print(
        f"Runtime    : "
        f"{runtime:.2f} seconds"
    )

    print(
        f"Checkpoint : "
        f"{best_checkpoint}"
    )

    return result


# ================================================================
# SAVE SUMMARY CSV WITHOUT PANDAS
# ================================================================

def save_summary_csv(
    results
):

    csv_path = (
        OUTPUT_DIR
        /
        "loss_ablation_summary.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([

            "Rank",

            "Loss",

            "F1",

            "IoU",

            "Precision",

            "Recall",

            "Accuracy",

            "Best Epoch",

            "Runtime Seconds"

        ])

        sorted_results = sorted(

            results,

            key=lambda x:
                x["best_f1"],

            reverse=True

        )

        for rank, result in enumerate(

            sorted_results,

            start=1

        ):

            writer.writerow([

                rank,

                result["loss"],

                f"{result['best_f1']:.6f}",

                f"{result['best_iou']:.6f}",

                f"{result['precision']:.6f}",

                f"{result['recall']:.6f}",

                f"{result['accuracy']:.6f}",

                result["best_epoch"],

                f"{result['runtime_seconds']:.2f}"

            ])

    print(
        f"\nCSV summary saved:\n{csv_path}"
    )


# ================================================================
# SAVE TEXT REPORT
# ================================================================

def save_report(
    results
):

    report_path = (
        OUTPUT_DIR
        /
        "loss_ablation_report.txt"
    )

    sorted_results = sorted(

        results,

        key=lambda x:
            x["best_f1"],

        reverse=True

    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "=" * 70
            + "\n"
        )

        file.write(
            "URBAN CHANGE DETECTION\n"
        )

        file.write(
            "LOSS FUNCTION ABLATION STUDY\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            f"Dataset root : "
            f"{DATASET_ROOT}\n"
        )

        file.write(
            f"Device       : "
            f"{get_device()}\n"
        )

        file.write(
            f"Image size   : "
            f"{IMAGE_SIZE}\n"
        )

        file.write(
            f"Batch size   : "
            f"{BATCH_SIZE}\n"
        )

        file.write(
            f"Epochs       : "
            f"{EPOCHS}\n"
        )

        file.write(
            f"Learning rate: "
            f"{LEARNING_RATE}\n"
        )

        file.write(
            f"Weight decay : "
            f"{WEIGHT_DECAY}\n"
        )

        file.write(
            f"Threshold    : "
            f"{THRESHOLD}\n\n"
        )

        file.write(
            "RESULTS\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        file.write(

            "Rank  Loss              "
            "F1        IoU       "
            "Precision  Recall\n"

        )

        file.write(
            "-" * 70
            + "\n"
        )

        for rank, result in enumerate(

            sorted_results,

            start=1

        ):

            file.write(

                f"{rank:<6}"

                f"{result['loss']:<18}"

                f"{result['best_f1']:<10.6f}"

                f"{result['best_iou']:<10.6f}"

                f"{result['precision']:<11.6f}"

                f"{result['recall']:<10.6f}"

                "\n"

            )

        best = sorted_results[0]

        file.write(
            "\n"
        )

        file.write(
            "=" * 70
            + "\n"
        )

        file.write(
            "BEST LOSS FUNCTION\n"
        )

        file.write(
            "=" * 70
            + "\n"
        )

        file.write(
            f"Loss      : "
            f"{best['loss']}\n"
        )

        file.write(
            f"Best F1   : "
            f"{best['best_f1']:.6f}\n"
        )

        file.write(
            f"Best IoU   : "
            f"{best['best_iou']:.6f}\n"
        )

        file.write(
            f"Precision  : "
            f"{best['precision']:.6f}\n"
        )

        file.write(
            f"Recall     : "
            f"{best['recall']:.6f}\n"
        )

        file.write(
            f"Accuracy   : "
            f"{best['accuracy']:.6f}\n"
        )

        file.write(
            f"Best epoch : "
            f"{best['best_epoch']}\n"
        )

    print(
        f"Report saved:\n{report_path}"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    start_time = time.time()

    set_seed(
        SEED
    )

    device = get_device()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "URBAN CHANGE DETECTION"
    )

    print(
        "LOSS FUNCTION ABLATION STUDY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDevice: {device}"
    )

    print(
        f"Dataset root: {DATASET_ROOT}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    # ------------------------------------------------------------
    # Check paths
    # ------------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CHECKING REQUIRED DIRECTORIES"
    )

    print(
        "=" * 60
    )

    check_directories()

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CREATING DATALOADERS"
    )

    print(
        "=" * 60
    )

    train_loader, val_loader = (
        create_dataloaders()
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STARTING LOSS ABLATION EXPERIMENTS"
    )

    print(
        "=" * 70
    )

    print(
        "\nLoss functions:"
    )

    for loss_name in LOSS_FUNCTIONS:

        print(
            f"  • {loss_name}"
        )

    results = []

    # ------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------

    for loss_name in LOSS_FUNCTIONS:

        set_seed(
            SEED
        )

        result = run_experiment(

            loss_name,

            train_loader,

            val_loader,

            device

        )

        results.append(
            result
        )

        # Save cumulative results

        save_json(

            results,

            OUTPUT_DIR
            /
            "all_results.json"

        )

    # ------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------

    sorted_results = sorted(

        results,

        key=lambda x:
            x["best_f1"],

        reverse=True

    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL LOSS ABLATION RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "\n"
        "Rank   Loss              "
        "F1        IoU       "
        "Precision   Recall"
    )

    print(
        "-" * 70
    )

    for rank, result in enumerate(

        sorted_results,

        start=1

    ):

        print(

            f"{rank:<7}"

            f"{result['loss']:<18}"

            f"{result['best_f1']:<10.6f}"

            f"{result['best_iou']:<10.6f}"

            f"{result['precision']:<12.6f}"

            f"{result['recall']:<10.6f}"

        )

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------

    save_summary_csv(
        results
    )

    save_report(
        results
    )

    save_json(

        {

            "dataset_root":
                str(DATASET_ROOT),

            "device":
                str(device),

            "image_size":
                IMAGE_SIZE,

            "batch_size":
                BATCH_SIZE,

            "epochs":
                EPOCHS,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "threshold":
                THRESHOLD,

            "results":
                sorted_results,

        },

        OUTPUT_DIR
        /
        "loss_ablation_final.json"

    )

    total_runtime = (
        time.time()
        -
        start_time
    )

    best = sorted_results[0]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOSS ABLATION COMPLETE 🎉🔥"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBest loss      : "
        f"{best['loss']}"
    )

    print(
        f"Best F1        : "
        f"{best['best_f1']:.6f}"
    )

    print(
        f"Best IoU       : "
        f"{best['best_iou']:.6f}"
    )

    print(
        f"Best Precision : "
        f"{best['precision']:.6f}"
    )

    print(
        f"Best Recall    : "
        f"{best['recall']:.6f}"
    )

    print(
        f"\nTotal runtime  : "
        f"{total_runtime:.2f} seconds"
    )

    print(
        f"\nResults directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nEverything completed successfully. 🎉🔥"
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()