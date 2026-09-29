"""
Training script for Urban Change Detection.
Uses the Canny/DSP dataset pipeline (dataset/levir_dataset.py) and the
ResNet34 + BAM + EdgeHead SiameseUNet, so training matches both the
proposal (Canny edge channel) and evaluate.py.
"""

import os
import csv
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from dataset import create_dataloaders
from models import SiameseUNet


def combined_loss(pred, target, edge_pred=None, edge_target=None, edge_weight=0.4):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred_sigmoid = torch.sigmoid(pred)
    smooth = 1.0
    intersection = (pred_sigmoid * target).sum()
    union = pred_sigmoid.sum() + target.sum()
    dice = 1 - (2.0 * intersection + smooth) / (union + smooth)
    loss = bce + dice
    if edge_pred is not None and edge_target is not None:
        edge_loss = F.binary_cross_entropy_with_logits(edge_pred, edge_target)
        loss = loss + edge_weight * edge_loss
    return loss


def calculate_metrics(preds, targets, threshold=0.45):
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = (targets > 0.5).float()
    
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()
    
    # --- ZERO-F1 MATH FIX ---
    if (tp + fp + fn) == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0
    # ------------------------

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    return accuracy, precision, recall, f1, iou

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_edge_target(mask):
    """Sobel edge map derived from the ground-truth mask, used only to
    supervise the model's auxiliary EdgeHead (not the DSP input channel)."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
    gx = F.conv2d(mask, sobel_x, padding=1)
    gy = F.conv2d(mask, sobel_y, padding=1)
    edges = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    return (edges > 0.5).float()


def main():
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders()
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = SiameseUNet(in_channels=4, pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    history_path = "training_history.csv"

    best_f1 = 0.0
    patience = 20
    counter = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            before = batch["before"].to(device)
            after = batch["after"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            change_logits, edge_logits = model(before, after, return_edges=True)
            edge_target = make_edge_target(mask)
            edge_target = F.interpolate(edge_target, size=edge_logits.shape[-2:], mode="nearest")

            loss = combined_loss(change_logits, mask, edge_logits, edge_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_acc = val_prec = val_rec = val_f1 = val_iou = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                before = batch["before"].to(device)
                after = batch["after"].to(device)
                mask = batch["mask"].to(device)

                change_logits = model(before, after)
                val_loss += combined_loss(change_logits, mask).item()

                acc, prec, rec, f1, iou = calculate_metrics(change_logits, mask)
                b = before.size(0)
                val_acc += acc * b
                val_prec += prec * b
                val_rec += rec * b
                val_f1 += f1 * b
                val_iou += iou * b
                total_samples += b

        val_loss /= len(val_loader)
        val_acc /= total_samples
        val_prec /= total_samples
        val_rec /= total_samples
        val_f1 /= total_samples
        val_iou /= total_samples

        print(f"Epoch {epoch}/{NUM_EPOCHS} | Time: {time.time()-start_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_accuracy": val_acc, "val_precision": val_prec, "val_recall": val_rec,
            "val_f1": val_f1, "val_iou": val_iou, "lr": scheduler.get_last_lr()[0],
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "metrics": {"f1": val_f1, "iou": val_iou},
            }, os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"New best model saved. F1 = {val_f1:.6f}")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"Training complete. Best F1: {best_f1:.6f}")


if __name__ == "__main__":
    main()