# Urban Change Detection with DSP and Deep Learning

**EEE 312 – Digital Signal Processing I Laboratory · BUET · January 2026 · Section B2, Group 04**

Mustasin Rahman (2206104) · Fahim Shahriyar (2206114) · Md. Ahasan Habib Kawsar (2206119) · Md. Kawsar Ahmed (2206122)

Detects new buildings between two satellite images of the same place (before / after). A classical DSP front-end (Gaussian low-pass filter + Canny edge detector) produces an edge map that is added to the RGB image as a 4th channel; a Siamese ResNet-34 U-Net with spatial-temporal attention (BAM), Siam-Concat fusion and an edge-guided decoder predicts the change map.

![Example](docs/example_test_102.jpg)

## Results (LEVIR-CD test set, 128 images, 1024 × 1024, dataset-level)

| Precision | Recall | F1 | IoU | Overall accuracy | Cohen's κ |
|---|---|---|---|---|---|
| 91.8 % | 90.6 % | **91.2 %** | **83.8 %** | 99.1 % | 0.907 |

Threshold 0.41 selected on the validation set; sliding-window inference (256-px tiles, 64-px overlap) with 4-view test-time augmentation.

## 1. Installation

```bash
git clone https://github.com/<your-account>/<repo-name>.git
cd <repo-name>
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10+ and PyTorch 2.x on macOS (Apple MPS). CUDA and CPU also work.

## 2. Trained model

GitHub does not store large files well, so the trained model is attached to the **Releases** page.

1. Download `best_model.pth` and `best_threshold.json` from *Releases → v2.0*.
2. Put both files in the `checkpoints/` folder.

## 3. Dataset (only needed for training / evaluation)

Download LEVIR-CD from the authors' page (https://justchenhao.github.io/LEVIR/) and arrange it as:

```
data/LEVIR-CD/{train,val,test}/{A,B,label}/*.png
```

## 4. User manual – GUI

```bash
python app.py
```

![GUI](docs/gui_screenshot.png)

1. **Load Before (T1)** and **Load After (T2)** – any two co-registered RGB images of the same area (PNG / JPG / TIFF).
2. *(Optional)* **Load Ground Truth** – a binary mask; the metrics panel then shows F1, IoU, precision and recall live.
3. Options: **Test-time augmentation** (more accurate, ~4× slower), **Post-processing** (removes speckles, straightens outlines into polygons), **Centre crop 512**, **Resolution** (choose the option matching your image's ground resolution; LEVIR-CD is 0.5 m/pixel).
4. Press **Run Inference**. The window stays responsive while the model runs.
5. Move the **Threshold** slider to trade false alarms against misses; **Validation Threshold** restores 0.41, **Auto Threshold (Otsu)** picks one per image.
6. **Save View** exports a six-panel PNG.

## 5. Command-line use

| Task | Command |
|---|---|
| Check the installation (1–3 min) | `python sanity_check.py` |
| Train (≈ 5 h on an Apple M-series laptop) | `python train.py` |
| Resume training | `python train.py --resume` |
| RGB-only ablation | `python train.py --no-canny` |
| Evaluate on the test set | `python evaluate.py --tta` |
| Per-image results for any folder with A/ and B/ | `python test_all_inference.py --data-dir <folder>` |
| DSP stage figures | `python visualization_slide.py data/LEVIR-CD/test/B/test_1.png` |

## 6. Project structure

```
config.py              all settings (paths, DSP parameters, training, threshold)
preprocessing/         Gaussian + Canny DSP front-end, channel standardisation
dataset/               LEVIR-CD dataset, random crops, synchronised augmentation
models/                ResNet-34 Siamese encoder, BAM + Siam-Concat fusion, decoder, edge head
losses/                BCE + Dice + class-balanced edge loss
training/              training loop, dataset-level metrics, threshold sweep
inference/             prediction on arbitrary image pairs, post-processing
evaluation/            figures and reports
gui/app.py             PyQt5 application (app.py is a launcher)
train.py, evaluate.py, sanity_check.py, test_all_inference.py
CHANGES_v2.md          list of bugs found in v1 and how they were fixed
presentation/          slides and video script
```

## 7. References

1. H. Chen and Z. Shi, "A spatial-temporal attention-based method and a new dataset for remote sensing image change detection," *Remote Sensing*, 12(10), 1662, 2020.
2. R. C. Daudt, B. Le Saux, A. Boulch, "Fully convolutional Siamese networks for change detection," ICIP 2018.
3. K. He et al., "Deep residual learning for image recognition," CVPR 2016.
4. O. Ronneberger, P. Fischer, T. Brox, "U-Net," MICCAI 2015.
5. J. Canny, "A computational approach to edge detection," IEEE TPAMI, 8(6), 1986.

The LEVIR-CD dataset belongs to its authors and is used for academic purposes only; it is not redistributed in this repository.
