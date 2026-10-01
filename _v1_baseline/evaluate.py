"""
============================================================
URBAN CHANGE DETECTION - TEST SET EVALUATION
============================================================
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import create_dataloaders
from models import SiameseUNet
from training.metrics import calculate_metrics
from evaluation.visualization import save_confusion_matrix, save_metric_summary
from evaluation.report import save_text_report, save_json_report

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")

NUM_VISUALIZATIONS = 20
THRESHOLD = 0.45  # Updated to optimal threshold

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    else: return torch.device("cpu")

def create_model(device):
    # Updated to correctly accept 4-channel DSP features
    model = SiameseUNet(in_channels=4) 
    model = model.to(device)
    return model

def load_model(model, device):
    print("\nLoading best model...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Best model loaded successfully.")
    return model

def create_output_directory():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def prepare_image_for_display(image):
    image = image.detach().cpu()
    image = image[:3] # Use first 3 channels for RGB
    image = image.permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)

def prepare_mask_for_display(mask):
    return mask.detach().cpu().squeeze().numpy()

def save_visualization(before, after, ground_truth, prediction, filename, sample_number):
    before_image = prepare_image_for_display(before)
    after_image = prepare_image_for_display(after)
    ground_truth_mask = prepare_mask_for_display(ground_truth)
    prediction_mask = prepare_mask_for_display(prediction)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(before_image); axes[0].set_title("Before"); axes[0].axis("off")
    axes[1].imshow(after_image); axes[1].set_title("After"); axes[1].axis("off")
    axes[2].imshow(ground_truth_mask, cmap="gray"); axes[2].set_title("Ground Truth"); axes[2].axis("off")
    axes[3].imshow(prediction_mask, cmap="gray"); axes[3].set_title("Prediction"); axes[3].axis("off")

    figure.suptitle(f"Urban Change Detection - {filename}", fontsize=14)
    figure.tight_layout()

    safe_filename = str(filename).replace("/", "_").replace("\\", "_").replace(" ", "_")
    output_path = os.path.join(RESULTS_DIR, f"{sample_number:03d}_{safe_filename}.png")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path

def evaluate_test_set(model, test_loader, device):
    print("\n" + "=" * 60 + "\nTEST SET EVALUATION\n" + "=" * 60)
    all_logits = []
    all_targets = []
    visualization_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            before = batch["before"].to(device)
            after = batch["after"].to(device)
            mask = batch["mask"].to(device)
            filenames = batch["filename"]

            logits = model(before, after)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= THRESHOLD).float()

            all_logits.append(logits.cpu())
            all_targets.append(mask.cpu())

            batch_size = before.shape[0]
            for i in range(batch_size):
                if visualization_count >= NUM_VISUALIZATIONS: break
                save_visualization(before[i], after[i], mask[i], predictions[i], filenames[i], visualization_count + 1)
                visualization_count += 1
            print(f"Processed batch {batch_index + 1}/{len(test_loader)}")

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics with the newly fixed function and optimal threshold
    metrics = calculate_metrics(all_logits, all_targets, threshold=THRESHOLD)
    return metrics

def main():
    print("=" * 60 + "\nURBAN CHANGE DETECTION - MODEL EVALUATION\n" + "=" * 60)
    device = get_device()
    create_output_directory()
    
    _, _, test_loader = create_dataloaders()
    model = create_model(device)
    model = load_model(model, device)
    
    metrics = evaluate_test_set(model, test_loader, device)

    print("\n" + "=" * 60 + "\nFINAL TEST RESULTS\n" + "=" * 60)
    print(f"Accuracy  : {metrics['accuracy']:.6f}")
    print(f"Precision : {metrics['precision']:.6f}")
    print(f"Recall    : {metrics['recall']:.6f}")
    print(f"F1-score  : {metrics['f1']:.6f}")
    print(f"IoU       : {metrics['iou']:.6f}\n")
    print(f"TP: {metrics['tp']} | TN: {metrics['tn']} | FP: {metrics['fp']} | FN: {metrics['fn']}")

    save_confusion_matrix(metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"], os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    save_metric_summary(metrics, os.path.join(RESULTS_DIR, "metrics_summary.png"))
    save_text_report(metrics, os.path.join(RESULTS_DIR, "test_metrics.txt"), test_samples=128)
    save_json_report(metrics, os.path.join(RESULTS_DIR, "test_metrics.json"), test_samples=128)
    print("\nEvaluation complete. ✅")

if __name__ == "__main__":
    main()