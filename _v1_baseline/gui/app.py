#!/usr/bin/env python3
"""
Urban Change Detection – PyQt5 GUI
Stage 15: Interactive Demonstration (with Morphological Post-Processing)
"""

import sys
import os
from pathlib import Path

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QGroupBox, QGridLayout,
    QStatusBar, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.siamese_unet import SiameseUNet
from preprocessing.dsp_processing import extract_dsp_features
from preprocessing.image_processing import load_rgb_image, resize_image

# Configuration
IMAGE_SIZE = 256
THRESHOLD_DEFAULT = 0.45
CROP_SIZE = 512
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class InferenceWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, before_path, after_path, device, crop_center=False):
        super().__init__()
        self.before_path = before_path
        self.after_path = after_path
        self.device = device
        self.crop_center = crop_center

    def load_and_preprocess(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.crop_center:
            h, w = img.shape[:2]
            size = min(CROP_SIZE, h, w)
            start_x = (w - size) // 2
            start_y = (h - size) // 2
            img = img[start_y:start_y+size, start_x:start_x+size]

        img = resize_image(img, IMAGE_SIZE)
        return img

    def run(self):
        try:
            before_rgb = self.load_and_preprocess(self.before_path)
            after_rgb = self.load_and_preprocess(self.after_path)

            before_4ch, _ = extract_dsp_features(before_rgb)
            after_4ch, _ = extract_dsp_features(after_rgb)

            before_tensor = torch.from_numpy(before_4ch).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            after_tensor = torch.from_numpy(after_4ch).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

            model = SiameseUNet(in_channels=4).to(self.device)
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.eval()

            with torch.no_grad():
                logits = model(before_tensor, after_tensor)
                prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

            before_display = before_rgb.astype(np.float32) / 255.0
            after_display = after_rgb.astype(np.float32) / 255.0

            result = {
                "before": before_display,
                "after": after_display,
                "probability": prob,
            }
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=3, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#2b2b2b')
        self.fig.patch.set_facecolor('#2b2b2b')
        self.axes.tick_params(colors='white')
        for spine in self.axes.spines.values():
            spine.set_color('white')
        self.setMinimumSize(200, 150)

    def plot_image(self, image, cmap=None, vmin=None, vmax=None, title=""):
        self.axes.clear()
        if image.ndim == 3:
            self.axes.imshow(image)
        else:
            im = self.axes.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            if cmap is not None:
                self.fig.colorbar(im, ax=self.axes, shrink=0.7)
        self.axes.set_title(title, color='white')
        self.axes.axis('off')
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Urban Change Detection – Siamese U-Net")
        self.setMinimumSize(1200, 800)

        self.before_path = None
        self.after_path = None
        self.gt_path = None
        self.probability = None
        self.before_img = None
        self.after_img = None
        self.gt_mask = None
        self.current_threshold = THRESHOLD_DEFAULT
        self.device = get_device()

        self.setup_ui()
        self.apply_dark_style()
        self.statusBar().showMessage("Ready. Load Before and After images.")

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Row 1: Load Buttons
        btn_layout = QHBoxLayout()
        self.btn_before = QPushButton("Load Before")
        self.btn_before.clicked.connect(self.load_before)
        self.btn_after = QPushButton("Load After")
        self.btn_after.clicked.connect(self.load_after)
        self.btn_gt = QPushButton("Load Ground Truth (optional)")
        self.btn_gt.clicked.connect(self.load_gt)
        btn_layout.addWidget(self.btn_before)
        btn_layout.addWidget(self.btn_after)
        btn_layout.addWidget(self.btn_gt)

        # Row 2: Analysis Tools
        tool_layout = QHBoxLayout()
        self.crop_checkbox = QCheckBox("Center Crop (512x512)")
        self.crop_checkbox.setChecked(False)  # Default OFF for 1024x1024 images
        self.btn_analyze = QPushButton("Run Inference")
        self.btn_analyze.clicked.connect(self.try_inference)
        self.btn_auto_thresh = QPushButton("Auto Threshold")
        self.btn_auto_thresh.clicked.connect(self.auto_threshold)
        self.btn_save = QPushButton("Save View")
        self.btn_save.clicked.connect(self.save_view)
        self.btn_save.setEnabled(False)

        tool_layout.addWidget(self.crop_checkbox)
        tool_layout.addWidget(self.btn_analyze)
        tool_layout.addWidget(self.btn_auto_thresh)
        tool_layout.addStretch()
        tool_layout.addWidget(self.btn_save)
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(tool_layout)

        # Image Display
        display_widget = QWidget()
        display_layout = QGridLayout(display_widget)

        self.label_before = QLabel("Before")
        self.label_before.setAlignment(Qt.AlignCenter)
        self.label_before.setStyleSheet("border: 1px solid #555; background: #1e1e1e;")
        self.label_before.setMinimumSize(300, 300)

        self.label_after = QLabel("After")
        self.label_after.setAlignment(Qt.AlignCenter)
        self.label_after.setStyleSheet("border: 1px solid #555; background: #1e1e1e;")
        self.label_after.setMinimumSize(300, 300)

        self.label_gt = QLabel("Ground Truth")
        self.label_gt.setAlignment(Qt.AlignCenter)
        self.label_gt.setStyleSheet("border: 1px solid #555; background: #1e1e1e;")
        self.label_gt.setMinimumSize(300, 300)

        display_layout.addWidget(self.label_before, 0, 0)
        display_layout.addWidget(self.label_after, 0, 1)
        display_layout.addWidget(self.label_gt, 0, 2)

        self.canvas_prob = MplCanvas(self, width=3, height=3)
        self.canvas_mask = MplCanvas(self, width=3, height=3)
        self.canvas_overlay = MplCanvas(self, width=3, height=3)
        display_layout.addWidget(self.canvas_prob, 1, 0)
        display_layout.addWidget(self.canvas_mask, 1, 1)
        display_layout.addWidget(self.canvas_overlay, 1, 2)
        main_layout.addWidget(display_widget)

        # Threshold Slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Threshold:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 99)
        self.slider.setValue(int(THRESHOLD_DEFAULT * 100))
        self.slider.setTickInterval(5)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.on_threshold_changed)
        self.label_threshold = QLabel(f"{THRESHOLD_DEFAULT:.2f}")
        self.label_threshold.setFixedWidth(50)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.label_threshold)
        main_layout.addLayout(slider_layout)

        # Metrics Panel
        metrics_group = QGroupBox("Metrics")
        metrics_layout = QHBoxLayout()
        self.lbl_f1 = QLabel("F1: --")
        self.lbl_iou = QLabel("IoU: --")
        self.lbl_prec = QLabel("Precision: --")
        self.lbl_recall = QLabel("Recall: --")
        self.lbl_acc = QLabel("Accuracy: --")
        self.lbl_area = QLabel("Change Area: --")
        for lbl in [self.lbl_f1, self.lbl_iou, self.lbl_prec, self.lbl_recall, self.lbl_acc, self.lbl_area]:
            lbl.setStyleSheet("font-weight: bold; color: #8ab4f8;")
        metrics_layout.addWidget(self.lbl_f1)
        metrics_layout.addWidget(self.lbl_iou)
        metrics_layout.addWidget(self.lbl_prec)
        metrics_layout.addWidget(self.lbl_recall)
        metrics_layout.addWidget(self.lbl_acc)
        metrics_layout.addWidget(self.lbl_area)
        metrics_group.setLayout(metrics_layout)
        main_layout.addWidget(metrics_group)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

    def apply_dark_style(self):
        style = """
            QMainWindow { background-color: #1e1e1e; }
            QWidget { background-color: #2b2b2b; color: #f0f0f0; font-family: Segoe UI; font-size: 10pt; }
            QPushButton { background-color: #3c3c3c; border: 1px solid #555; border-radius: 5px; padding: 6px 12px; }
            QPushButton:hover { background-color: #505050; }
            QPushButton:pressed { background-color: #6a6a6a; }
            QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 0.5em; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLabel { color: #f0f0f0; }
            QSlider::groove:horizontal { border: 1px solid #555; height: 8px; background: #3c3c3c; border-radius: 4px; }
            QSlider::handle:horizontal { background: #8ab4f8; border: 1px solid #5a7a9a; width: 18px; margin: -5px 0; border-radius: 9px; }
            QSlider::handle:horizontal:hover { background: #a0c8ff; }
            QStatusBar { background-color: #1e1e1e; color: #aaa; }
            QCheckBox { color: #f0f0f0; }
        """
        self.setStyleSheet(style)

    def load_before(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Before Image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if path:
            self.before_path = path
            self.status.showMessage(f"Loaded Before: {os.path.basename(path)}")

    def load_after(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select After Image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if path:
            self.after_path = path
            self.status.showMessage(f"Loaded After: {os.path.basename(path)}")

    def load_gt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Ground Truth", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if path:
            self.gt_path = path
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                self.gt_mask = (mask > 127).astype(np.float32)
                self.display_gt()
                self.update_metrics()
                self.status.showMessage(f"Loaded GT: {os.path.basename(path)}")

    def try_inference(self):
        if not self.before_path or not self.after_path:
            QMessageBox.warning(self, "Error", "Please load both Before and After images.")
            return
        
        self.btn_analyze.setEnabled(False)
        self.status.showMessage("Running inference...")
        
        crop_center = self.crop_checkbox.isChecked()
        self.worker = InferenceWorker(self.before_path, self.after_path, self.device, crop_center)
        self.worker.finished.connect(self.on_inference_done)
        self.worker.error.connect(self.on_inference_error)
        self.worker.start()

    def on_inference_done(self, result):
        self.before_img = result["before"]
        self.after_img = result["after"]
        self.probability = result["probability"]
        
        self.display_before_after()
        self.display_probability()
        self.update_binary_and_overlay()
        self.btn_save.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.status.showMessage("Inference complete.")

    def on_inference_error(self, msg):
        self.btn_analyze.setEnabled(True)
        self.status.showMessage("Error: " + msg)
        QMessageBox.critical(self, "Inference Error", msg)

    def auto_threshold(self):
        if self.probability is None:
            QMessageBox.warning(self, "Error", "Run inference first.")
            return
        
        prob = self.probability
        mean = prob.mean()
        std = prob.std()
        
        suggested = max(0.05, mean + std)
        if mean > 0.3:
            suggested = max(0.4, min(0.55, suggested))
        
        val = int(suggested * 100)
        self.slider.setValue(val)
        self.current_threshold = suggested
        self.label_threshold.setText(f"{suggested:.2f}")
        self.update_binary_and_overlay()
        self.status.showMessage(f"Auto Threshold set to {suggested:.2f} (Mean={mean:.3f}, Std={std:.3f})")

    def display_before_after(self):
        self.display_image_on_label(self.label_before, self.before_img)
        self.display_image_on_label(self.label_after, self.after_img)

    def display_gt(self):
        if self.gt_mask is not None:
            self.display_image_on_label(self.label_gt, self.gt_mask, is_gray=True)

    def display_image_on_label(self, label, image, is_gray=False):
        if image is None: return
        if is_gray:
            img_uint8 = (image * 255).astype(np.uint8)
            h, w = image.shape
            qimg = QImage(img_uint8.data, w, h, w, QImage.Format_Grayscale8)
        else:
            img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            if image.ndim == 3:
                h, w, c = image.shape
                qimg = QImage(img_uint8.data, w, h, w * c, QImage.Format_RGB888)
            else:
                h, w = image.shape
                qimg = QImage(img_uint8.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def display_probability(self):
        if self.probability is not None:
            self.canvas_prob.plot_image(self.probability, cmap='viridis', vmin=0, vmax=1, title="Probability")

    def get_cleaned_binary(self):
        """Applies Morphological Post-Processing to smooth boundaries and remove speckles."""
        binary = (self.probability >= self.current_threshold).astype(np.float32)
        # 5x5 kernel for mathematical morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # 1. Opening: Removes isolated false positive pixels (noise)
        binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        # 2. Closing: Fills in small black holes inside detected buildings (false negatives)
        binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_CLOSE, kernel)
        return binary_cleaned

    def update_binary_and_overlay(self):
        if self.probability is None: return
        threshold = self.current_threshold
        
        # Get the smoothed out binary mask
        binary_cleaned = self.get_cleaned_binary()
        
        self.canvas_mask.plot_image(binary_cleaned, cmap='gray', vmin=0, vmax=1, title=f"Binary (T={threshold:.2f})")
        
        overlay = self.before_img.copy()
        if overlay.ndim == 3:
            r, g, b = overlay[:,:,0], overlay[:,:,1], overlay[:,:,2]
            r[binary_cleaned > 0.5] = 1.0
            g[binary_cleaned > 0.5] = 0.0
            b[binary_cleaned > 0.5] = 0.0
            overlay = np.stack([r, g, b], axis=-1)
        self.canvas_overlay.plot_image(overlay, title=f"Overlay (T={threshold:.2f})")
        
        self.update_metrics(binary_cleaned)

    def on_threshold_changed(self):
        self.current_threshold = self.slider.value() / 100.0
        self.label_threshold.setText(f"{self.current_threshold:.2f}")
        self.update_binary_and_overlay()

    def update_metrics(self, pred_mask=None):
        if self.probability is None or self.gt_mask is None:
            return
            
        if pred_mask is None:
            pred = self.get_cleaned_binary()
        else:
            pred = pred_mask

        gt = self.gt_mask
        tp = np.logical_and(pred == 1, gt == 1).sum()
        tn = np.logical_and(pred == 0, gt == 0).sum()
        fp = np.logical_and(pred == 1, gt == 0).sum()
        fn = np.logical_and(pred == 0, gt == 1).sum()
        
        # --- ZERO-F1 MATH FIX ---
        if (tp + fp + fn) == 0:
            acc, prec, rec, f1, iou = 1.0, 1.0, 1.0, 1.0, 1.0
        else:
            eps = 1e-8
            acc = (tp + tn) / (tp + tn + fp + fn + eps)
            prec = tp / (tp + fp + eps)
            rec = tp / (tp + fn + eps)
            f1 = 2 * prec * rec / (prec + rec + eps)
            iou = tp / (tp + fp + fn + eps)
            
        area = (gt.sum() / gt.size) * 100
        
        self.lbl_f1.setText(f"F1: {f1:.4f}")
        self.lbl_iou.setText(f"IoU: {iou:.4f}")
        self.lbl_prec.setText(f"Precision: {prec:.4f}")
        self.lbl_recall.setText(f"Recall: {rec:.4f}")
        self.lbl_acc.setText(f"Accuracy: {acc:.4f}")
        self.lbl_area.setText(f"Change Area: {area:.2f}%")

    def save_view(self):
        if self.probability is None: return
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        axes[0].imshow(self.before_img); axes[0].set_title("Before"); axes[0].axis('off')
        axes[1].imshow(self.after_img); axes[1].set_title("After"); axes[1].axis('off')
        if self.gt_mask is not None:
            axes[2].imshow(self.gt_mask, cmap='gray')
        else:
            axes[2].text(0.5, 0.5, "No GT", ha='center', va='center')
        axes[2].axis('off')
        
        axes[3].imshow(self.probability, cmap='viridis', vmin=0, vmax=1); axes[3].set_title("Probability"); axes[3].axis('off')
        
        binary_cleaned = self.get_cleaned_binary()
        axes[4].imshow(binary_cleaned, cmap='gray'); axes[4].set_title(f"Binary (T={self.current_threshold:.2f})"); axes[4].axis('off')
        
        overlay = self.before_img.copy()
        if overlay.ndim == 3:
            r, g, b = overlay[:,:,0], overlay[:,:,1], overlay[:,:,2]
            r[binary_cleaned > 0.5] = 1.0; g[binary_cleaned > 0.5] = 0.0; b[binary_cleaned > 0.5] = 0.0
            overlay = np.stack([r, g, b], axis=-1)
        axes[5].imshow(overlay); axes[5].set_title("Overlay"); axes[5].axis('off')
        plt.tight_layout()
        
        path, _ = QFileDialog.getSaveFileName(self, "Save View", "change_detection_view.png", "PNG (*.png)")
        if path:
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.status.showMessage(f"Saved to {path}")
        else:
            plt.close(fig)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()