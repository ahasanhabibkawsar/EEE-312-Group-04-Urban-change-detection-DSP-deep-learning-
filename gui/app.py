#!/usr/bin/env python3
"""
Urban Change Detection - PyQt5 GUI (v2)

Interactive demonstration of the trained Siamese U-Net.

* The model is loaded ONCE (in a background thread) and reused.
* Inference runs in a QThread, so the window never freezes.
* Images are predicted at native resolution with the sliding window
  (no down-sizing to 256 x 256).
* Optional test-time augmentation (4 flips), optional resampling for
  imagery whose ground resolution differs from LEVIR-CD's 0.5 m/pixel.
* Threshold slider re-thresholds instantly (no new inference).
* Optional post-processing: morphological opening, small-region removal
  and polygon approximation (cv2.approxPolyDP) for crisp footprints.
* If a ground-truth mask is loaded, F1 / IoU / Precision / Recall are
  computed live for the mask shown on screen.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QGroupBox, QGridLayout,
    QStatusBar, QMessageBox, QCheckBox, QComboBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from inference.predictor import (  # noqa: E402
    get_device, load_model, run_inference, postprocess_mask,
    center_crop, rescale,
)
from training.metrics import metrics_from_counts  # noqa: E402


CROP_SIZE = 512
SCALE_OPTIONS = [
    ("Native resolution (0.5 m/px, LEVIR-CD)", 1.0),
    ("Source 1.0 m/px  → upsample ×2", 2.0),
    ("Source 0.3 m/px  → downsample ×0.6", 0.6),
    ("Source 0.25 m/px → downsample ×0.5", 0.5),
]


# =========================================================
# Worker threads
# =========================================================

class ModelLoader(QThread):
    loaded = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, device):
        super().__init__()
        self.device = device

    def run(self):
        try:
            self.loaded.emit(load_model(self.device, config.CHECKPOINT_PATH))
        except Exception as exc:  # shown in a dialog
            self.error.emit(str(exc))


class InferenceWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model, before_path, after_path, device, crop, tta, scale):
        super().__init__()
        self.model = model
        self.before_path = before_path
        self.after_path = after_path
        self.device = device
        self.crop = crop
        self.tta = tta
        self.scale = scale

    def run(self):
        try:
            result = run_inference(
                self.model, self.before_path, self.after_path, self.device,
                tta=self.tta,
                crop_size=CROP_SIZE if self.crop else None,
                scale=self.scale,
            )
            result["before"] = result["before"].astype(np.float32) / 255.0
            result["after"] = result["after"].astype(np.float32) / 255.0
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# =========================================================
# Matplotlib canvas
# =========================================================

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=3, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor("#2b2b2b")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumSize(220, 180)

    def plot_image(self, image, cmap=None, vmin=None, vmax=None, title="", colorbar=False):
        # Clearing the whole figure also removes the previous colour bar
        # (v1 added a new colour bar on every redraw).
        self.fig.clear()
        axes = self.fig.add_subplot(111)
        axes.set_facecolor("#2b2b2b")

        im = axes.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        if colorbar:
            bar = self.fig.colorbar(im, ax=axes, shrink=0.75)
            bar.ax.tick_params(colors="white")

        axes.set_title(title, color="white", fontsize=10)
        axes.axis("off")
        self.fig.tight_layout()
        self.draw()


# =========================================================
# Main window
# =========================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Urban Change Detection – Siamese U-Net (DSP + Deep Learning)")
        self.setMinimumSize(1250, 850)

        self.device = get_device()
        self.model = None
        self.worker = None

        self.before_path = None
        self.after_path = None
        self.gt_path = None

        self.before_img = None
        self.after_img = None
        self.probability = None
        self.gt_mask = None

        self.validation_threshold = config.load_best_threshold()
        self.current_threshold = self.validation_threshold

        self.setup_ui()
        self.apply_dark_style()

        self.status.showMessage(f"Loading model on {self.device} ...")
        self.btn_analyze.setEnabled(False)
        self.loader = ModelLoader(self.device)
        self.loader.loaded.connect(self.on_model_loaded)
        self.loader.error.connect(self.on_model_error)
        self.loader.start()

    # -----------------------------------------------------
    # UI
    # -----------------------------------------------------

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Row 1: file buttons
        file_row = QHBoxLayout()
        self.btn_before = QPushButton("Load Before (T1)")
        self.btn_after = QPushButton("Load After (T2)")
        self.btn_gt = QPushButton("Load Ground Truth (optional)")
        self.btn_before.clicked.connect(self.load_before)
        self.btn_after.clicked.connect(self.load_after)
        self.btn_gt.clicked.connect(self.load_gt)
        for button in (self.btn_before, self.btn_after, self.btn_gt):
            file_row.addWidget(button)
        main_layout.addLayout(file_row)

        # Row 2: options
        option_row = QHBoxLayout()
        self.crop_checkbox = QCheckBox(f"Centre crop ({CROP_SIZE}×{CROP_SIZE})")
        self.tta_checkbox = QCheckBox("Test-time augmentation")
        self.tta_checkbox.setChecked(True)
        self.post_checkbox = QCheckBox("Post-processing (opening + polygons)")
        self.post_checkbox.setChecked(True)
        self.post_checkbox.stateChanged.connect(self.update_binary_and_overlay)
        self.scale_combo = QComboBox()
        for label, _ in SCALE_OPTIONS:
            self.scale_combo.addItem(label)

        for widget in (self.crop_checkbox, self.tta_checkbox, self.post_checkbox):
            option_row.addWidget(widget)
        option_row.addWidget(QLabel("Resolution:"))
        option_row.addWidget(self.scale_combo)
        option_row.addStretch()
        main_layout.addLayout(option_row)

        # Row 3: actions
        action_row = QHBoxLayout()
        self.btn_analyze = QPushButton("Run Inference")
        self.btn_analyze.clicked.connect(self.try_inference)
        self.btn_auto_thresh = QPushButton("Auto Threshold (Otsu)")
        self.btn_auto_thresh.clicked.connect(self.auto_threshold)
        self.btn_val_thresh = QPushButton(f"Validation Threshold ({self.validation_threshold:.2f})")
        self.btn_val_thresh.clicked.connect(lambda: self.set_threshold(self.validation_threshold))
        self.btn_save = QPushButton("Save View")
        self.btn_save.clicked.connect(self.save_view)
        self.btn_save.setEnabled(False)
        for button in (self.btn_analyze, self.btn_auto_thresh, self.btn_val_thresh):
            action_row.addWidget(button)
        action_row.addStretch()
        action_row.addWidget(self.btn_save)
        main_layout.addLayout(action_row)

        # Image grid
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)

        self.label_before = self._image_label("Before")
        self.label_after = self._image_label("After")
        self.label_gt = self._image_label("Ground Truth")
        grid.addWidget(self.label_before, 0, 0)
        grid.addWidget(self.label_after, 0, 1)
        grid.addWidget(self.label_gt, 0, 2)

        self.canvas_prob = MplCanvas(self)
        self.canvas_mask = MplCanvas(self)
        self.canvas_overlay = MplCanvas(self)
        grid.addWidget(self.canvas_prob, 1, 0)
        grid.addWidget(self.canvas_mask, 1, 1)
        grid.addWidget(self.canvas_overlay, 1, 2)
        main_layout.addWidget(grid_widget)

        # Threshold slider (0.01 steps)
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Threshold:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 99)
        self.slider.setValue(int(round(self.current_threshold * 100)))
        self.slider.setTickInterval(5)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.on_threshold_changed)
        self.label_threshold = QLabel(f"{self.current_threshold:.2f}")
        self.label_threshold.setFixedWidth(50)
        slider_row.addWidget(self.slider)
        slider_row.addWidget(self.label_threshold)
        main_layout.addLayout(slider_row)

        # Metrics
        metrics_group = QGroupBox("Metrics (vs. ground truth, for the mask shown)")
        metrics_row = QHBoxLayout()
        self.lbl_f1 = QLabel("F1: --")
        self.lbl_iou = QLabel("IoU: --")
        self.lbl_prec = QLabel("Precision: --")
        self.lbl_recall = QLabel("Recall: --")
        self.lbl_acc = QLabel("Accuracy: --")
        self.lbl_area = QLabel("Predicted change: --")
        self.lbl_gt_area = QLabel("True change: --")
        for label in (self.lbl_f1, self.lbl_iou, self.lbl_prec, self.lbl_recall,
                      self.lbl_acc, self.lbl_area, self.lbl_gt_area):
            label.setStyleSheet("font-weight: bold; color: #8ab4f8;")
            metrics_row.addWidget(label)
        metrics_group.setLayout(metrics_row)
        main_layout.addWidget(metrics_group)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

    @staticmethod
    def _image_label(text):
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("border: 1px solid #555; background: #1e1e1e;")
        label.setMinimumSize(300, 300)
        return label

    def apply_dark_style(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { background-color: #2b2b2b; color: #f0f0f0; font-size: 10pt; }
            QPushButton { background-color: #3c3c3c; border: 1px solid #555; border-radius: 5px; padding: 6px 12px; }
            QPushButton:hover { background-color: #505050; }
            QPushButton:pressed { background-color: #6a6a6a; }
            QPushButton:disabled { color: #777; }
            QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 0.6em; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QSlider::groove:horizontal { border: 1px solid #555; height: 8px; background: #3c3c3c; border-radius: 4px; }
            QSlider::handle:horizontal { background: #8ab4f8; border: 1px solid #5a7a9a; width: 18px; margin: -5px 0; border-radius: 9px; }
            QStatusBar { background-color: #1e1e1e; color: #aaa; }
            QComboBox { background-color: #3c3c3c; border: 1px solid #555; padding: 3px; }
        """)

    # -----------------------------------------------------
    # Model loading
    # -----------------------------------------------------

    def on_model_loaded(self, model):
        self.model = model
        self.btn_analyze.setEnabled(True)
        channels = "RGB + Canny" if model.in_channels == 4 else "RGB only"
        self.status.showMessage(f"Model ready on {self.device} ({channels}). Load Before and After images.")

    def on_model_error(self, message):
        self.status.showMessage("Model could not be loaded.")
        QMessageBox.critical(self, "Model Error", message)

    # -----------------------------------------------------
    # File loading
    # -----------------------------------------------------

    def _pick(self, title):
        path, _ = QFileDialog.getOpenFileName(
            self, title, "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        return path

    def load_before(self):
        path = self._pick("Select Before Image")
        if path:
            self.before_path = path
            self.status.showMessage(f"Loaded Before: {os.path.basename(path)}")

    def load_after(self):
        path = self._pick("Select After Image")
        if path:
            self.after_path = path
            self.status.showMessage(f"Loaded After: {os.path.basename(path)}")

    def load_gt(self):
        path = self._pick("Select Ground Truth")
        if path:
            self.gt_path = path
            self.status.showMessage(f"Loaded GT: {os.path.basename(path)}")
            if self.probability is not None:
                self.prepare_gt()
                self.display_gt()
                self.update_binary_and_overlay()

    def prepare_gt(self):
        """Load the GT with the SAME crop / scale as the images."""

        self.gt_mask = None
        if not self.gt_path:
            return

        mask = cv2.imread(self.gt_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return

        if self.crop_checkbox.isChecked():
            mask = center_crop(mask, CROP_SIZE)
        mask = rescale(mask, SCALE_OPTIONS[self.scale_combo.currentIndex()][1], is_mask=True)

        if self.probability is not None and mask.shape != self.probability.shape:
            h, w = self.probability.shape
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        self.gt_mask = (mask > 127).astype(np.float32)

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    def try_inference(self):
        if self.model is None:
            QMessageBox.warning(self, "Please wait", "The model is still loading.")
            return
        if not self.before_path or not self.after_path:
            QMessageBox.warning(self, "Error", "Please load both Before and After images.")
            return

        self.btn_analyze.setEnabled(False)
        tta = self.tta_checkbox.isChecked()
        self.status.showMessage("Running inference" + (" with TTA" if tta else "") + " ...")

        self.worker = InferenceWorker(
            self.model, self.before_path, self.after_path, self.device,
            crop=self.crop_checkbox.isChecked(),
            tta=tta,
            scale=SCALE_OPTIONS[self.scale_combo.currentIndex()][1],
        )
        self.worker.finished.connect(self.on_inference_done)
        self.worker.error.connect(self.on_inference_error)
        self.worker.start()

    def on_inference_done(self, result):
        self.before_img = result["before"]
        self.after_img = result["after"]
        self.probability = result["probability"]

        self.prepare_gt()
        self.display_image_on_label(self.label_before, self.before_img)
        self.display_image_on_label(self.label_after, self.after_img)
        self.display_gt()
        self.canvas_prob.plot_image(self.probability, cmap="viridis", vmin=0, vmax=1,
                                    title="Change probability", colorbar=True)
        self.update_binary_and_overlay()

        self.btn_save.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        h, w = self.probability.shape
        self.status.showMessage(f"Inference complete ({w}×{h} px).")

    def on_inference_error(self, message):
        self.btn_analyze.setEnabled(True)
        self.status.showMessage("Error: " + message)
        QMessageBox.critical(self, "Inference Error", message)

    # -----------------------------------------------------
    # Threshold
    # -----------------------------------------------------

    def set_threshold(self, value):
        self.slider.setValue(int(round(value * 100)))   # triggers on_threshold_changed

    def on_threshold_changed(self):
        self.current_threshold = self.slider.value() / 100.0
        self.label_threshold.setText(f"{self.current_threshold:.2f}")
        self.update_binary_and_overlay()

    def auto_threshold(self):
        """Otsu's method on the probability histogram (per image)."""

        if self.probability is None:
            QMessageBox.warning(self, "Error", "Run inference first.")
            return

        prob_u8 = np.clip(self.probability * 255, 0, 255).astype(np.uint8)
        otsu, _ = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        suggested = float(np.clip(otsu / 255.0, 0.05, 0.95))

        self.set_threshold(suggested)
        self.status.showMessage(f"Otsu threshold for this image: {suggested:.2f} "
                                f"(validation threshold: {self.validation_threshold:.2f})")

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    def display_gt(self):
        if self.gt_mask is not None:
            self.display_image_on_label(self.label_gt, self.gt_mask, is_gray=True)

    def display_image_on_label(self, label, image, is_gray=False):
        if image is None:
            return
        if is_gray or image.ndim == 2:
            img = np.ascontiguousarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            img = np.ascontiguousarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
            h, w, c = img.shape
            qimg = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def get_display_mask(self):
        binary = (self.probability >= self.current_threshold).astype(np.float32)
        if self.post_checkbox.isChecked():
            binary = postprocess_mask(binary, min_area=20, open_kernel=3, polygonize=True)
        return binary

    def make_overlay(self, mask):
        """Predicted change drawn on the AFTER image (red fill + outline)."""

        overlay = self.after_img.copy()
        changed = mask > 0.5
        overlay[changed] = 0.55 * overlay[changed] + 0.45 * np.array([1.0, 0.0, 0.0])

        contours, _ = cv2.findContours(changed.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        overlay_u8 = np.ascontiguousarray((overlay * 255).astype(np.uint8))
        cv2.drawContours(overlay_u8, contours, -1, (255, 255, 0), 1)
        return overlay_u8.astype(np.float32) / 255.0

    def update_binary_and_overlay(self):
        if self.probability is None:
            return

        mask = self.get_display_mask()
        t = self.current_threshold
        self.canvas_mask.plot_image(mask, cmap="gray", vmin=0, vmax=1, title=f"Change mask (t = {t:.2f})")
        self.canvas_overlay.plot_image(self.make_overlay(mask), title="Detected change on After image")
        self.update_metrics(mask)

    def update_metrics(self, mask):
        area = 100.0 * float(mask.mean())
        self.lbl_area.setText(f"Predicted change: {area:.2f}%")

        if self.gt_mask is None or self.gt_mask.shape != mask.shape:
            for label, name in ((self.lbl_f1, "F1"), (self.lbl_iou, "IoU"), (self.lbl_prec, "Precision"),
                                (self.lbl_recall, "Recall"), (self.lbl_acc, "Accuracy")):
                label.setText(f"{name}: --")
            self.lbl_gt_area.setText("True change: --")
            return

        pred = mask > 0.5
        gt = self.gt_mask > 0.5
        m = metrics_from_counts(
            np.logical_and(pred, gt).sum(), np.logical_and(~pred, ~gt).sum(),
            np.logical_and(pred, ~gt).sum(), np.logical_and(~pred, gt).sum(),
        )

        self.lbl_f1.setText(f"F1: {m['f1']:.4f}")
        self.lbl_iou.setText(f"IoU: {m['iou']:.4f}")
        self.lbl_prec.setText(f"Precision: {m['precision']:.4f}")
        self.lbl_recall.setText(f"Recall: {m['recall']:.4f}")
        self.lbl_acc.setText(f"Accuracy: {m['accuracy']:.4f}")
        self.lbl_gt_area.setText(f"True change: {100.0 * float(gt.mean()):.2f}%")

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    def save_view(self):
        if self.probability is None:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save View", "change_detection_view.png", "PNG (*.png)")
        if not path:
            return

        mask = self.get_display_mask()
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        axes[0].imshow(self.before_img); axes[0].set_title("Before")
        axes[1].imshow(self.after_img); axes[1].set_title("After")
        if self.gt_mask is not None:
            axes[2].imshow(self.gt_mask, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Ground truth")
        else:
            axes[2].text(0.5, 0.5, "No ground truth", ha="center", va="center")
        axes[3].imshow(self.probability, cmap="viridis", vmin=0, vmax=1); axes[3].set_title("Probability")
        axes[4].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[4].set_title(f"Change mask (t = {self.current_threshold:.2f})")
        axes[5].imshow(self.make_overlay(mask)); axes[5].set_title("Overlay")
        for axis in axes:
            axis.axis("off")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.status.showMessage(f"Saved to {path}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
