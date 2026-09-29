"""
============================================================
URBAN CHANGE DETECTION - FULL TEST-SET INFERENCE
============================================================
"""
import cv2
import os
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset.levir_dataset import LEVIRCDDataset
from models.siamese_unet import SiameseUNet

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "data" / "LEVIR-CD"

TEST_A_DIR = DATASET_ROOT / "test" / "A"
TEST_B_DIR = DATASET_ROOT / "test" / "B"
TEST_LABEL_DIR = DATASET_ROOT / "test" / "label"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

RESULTS_DIR = PROJECT_ROOT / "full_inference_results"
VISUALIZATION_DIR = RESULTS_DIR / "visualizations"
CSV_PATH = RESULTS_DIR / "per_image_metrics.csv"
JSON_PATH = RESULTS_DIR / "overall_metrics.json"

IMAGE_SIZE = 256
THRESHOLD = 0.45  # Updated to optimal threshold

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def create_result_directories():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

def create_test_dataset():
    return LEVIRCDDataset(
        image_a_dir=TEST_A_DIR, image_b_dir=TEST_B_DIR, label_dir=TEST_LABEL_DIR,
        image_size=IMAGE_SIZE, augment=False,
    )

def create_model(device):
    # Ensure 4-channel input
    model = SiameseUNet(in_channels=4).to(device)
    return model

def load_checkpoint(model, device):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def calculate_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(prediction, target).sum()
    tn = np.logical_and(~prediction, ~target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()
    
    tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)

    # --- THE ZERO-F1 MATH FIX ---
    if (tp + fp + fn) == 0:
        return {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def prepare_display_image(image):
    image = image.detach().cpu().numpy()
    if image.ndim == 4: image = image[0]
    if image.shape[0] >= 3:
        image = image[:3]
        image = np.transpose(image, (1, 2, 0))
    else:
        image = image[0]
    return np.clip(image, 0.0, 1.0)

def save_visualization(before, after, target, prediction, filename, metrics):
    before_img = prepare_display_image(before)
    after_img = prepare_display_image(after)
    target = target.squeeze().detach().cpu().numpy()
    prediction = prediction.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(before_img); axes[0].set_title("Before"); axes[0].axis("off")
    axes[1].imshow(after_img); axes[1].set_title("After"); axes[1].axis("off")
    axes[2].imshow(target, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Ground Truth"); axes[2].axis("off")
    axes[3].imshow(prediction, cmap="gray", vmin=0, vmax=1); axes[3].set_title(f"Prediction\nF1={metrics['f1']:.3f}"); axes[3].axis("off")

    fig.suptitle(filename, fontsize=14)
    plt.tight_layout()
    output_path = VISUALIZATION_DIR / (Path(filename).stem + "_inference.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def run_inference(model, dataset, device):
    print("\n" + "=" * 60 + "\nRUNNING FULL TEST-SET INFERENCE (WITH MORPHOLOGY)\n" + "=" * 60)
    results = []
    total_start_time = time.time()
    
    # Define the 5x5 kernel for mathematical morphology cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            before = sample["before"]
            after = sample["after"]
            mask = sample["mask"]
            filename = sample.get("filename", f"sample_{index}.png")

            before_batch = before.unsqueeze(0).to(device)
            after_batch = after.unsqueeze(0).to(device)
            mask_batch = mask.unsqueeze(0)

            # Standard Inference
            logits = model(before_batch, after_batch)
            probabilities = torch.sigmoid(logits)
            
            # Convert to numpy array for OpenCV processing
            pred_mask = (probabilities[0, 0].cpu().numpy() >= THRESHOLD).astype(np.uint8)
            
            # --- MORPHOLOGICAL POST-PROCESSING ---
            # 1. Opening: Removes isolated false positive pixels (noise)
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
            # 2. Closing: Fills in small black holes inside detected buildings
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
            # -------------------------------------

            metrics = calculate_metrics(
                pred_mask,
                mask_batch.squeeze(0).squeeze(0).numpy()
            )

            total_pixels = pred_mask.size
            predicted_positive = int(pred_mask.sum())
            actual_positive = int(mask_batch.sum().item())

            result = {
                "filename": filename,
                **metrics,
                "predicted_pixels": predicted_positive,
                "actual_pixels": actual_positive,
                "predicted_change_percent": (predicted_positive / total_pixels * 100),
                "actual_change_percent": (actual_positive / total_pixels * 100),
            }
            results.append(result)

            # Ensure the tensor is formatted correctly for save_visualization
            prediction_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0)

            if index < 10:  # Save first 10 for quick visual check
                save_visualization(before, after, mask, prediction_tensor.squeeze(0), filename, metrics)

            print(f"Processed {index + 1:3d}/{len(dataset)} | {filename:<20} | F1: {metrics['f1']:.4f}")

    elapsed = time.time() - total_start_time
    print(f"\nTotal inference time: {elapsed:.2f}s")
    return results

def calculate_overall_metrics(results):
    overall = {}
    for name in ["accuracy", "precision", "recall", "f1", "iou"]:
        overall[name] = float(np.mean([r[name] for r in results]))

    total_tp = sum(r["tp"] for r in results)
    total_tn = sum(r["tn"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    total_pixels = total_tp + total_tn + total_fp + total_fn

    overall["micro_accuracy"] = (total_tp + total_tn) / total_pixels if total_pixels > 0 else 0
    overall["micro_precision"] = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    overall["micro_recall"] = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    
    p, r = overall["micro_precision"], overall["micro_recall"]
    overall["micro_f1"] = 2 * p * r / (p + r) if p + r > 0 else 0
    overall["micro_iou"] = total_tp / (total_tp + total_fp + total_fn) if total_tp + total_fp + total_fn > 0 else 0
    
    overall.update({"tp": total_tp, "tn": total_tn, "fp": total_fp, "fn": total_fn})
    return overall

def save_csv(results):
    fieldnames = ["filename", "accuracy", "precision", "recall", "f1", "iou", "tp", "tn", "fp", "fn", 
                  "predicted_pixels", "actual_pixels", "predicted_change_percent", "actual_change_percent"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def save_json(overall):
    with open(JSON_PATH, "w") as f:
        json.dump(overall, f, indent=4)

def main():
    device = get_device()
    create_result_directories()
    dataset = create_test_dataset()
    model = create_model(device)
    model = load_checkpoint(model, device)
    
    results = run_inference(model, dataset, device)
    overall = calculate_overall_metrics(results)
    
    save_csv(results)
    save_json(overall)
    
    print("\n" + "=" * 60 + "\nFINAL FULL TEST-SET RESULTS\n" + "=" * 60)
    print(f"Macro F1-score  : {overall['f1']:.6f}")
    print(f"Micro F1-score  : {overall['micro_f1']:.6f}")
    print("FULL INFERENCE COMPLETE ✅")

if __name__ == "__main__":
    main()