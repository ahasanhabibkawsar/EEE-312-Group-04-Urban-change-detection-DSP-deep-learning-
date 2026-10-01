# Urban Change Detection — v2 changes

v1 code, checkpoint, training history and test metrics are preserved in `_v1_baseline/`
(v1 test result: F1 0.7711, IoU 0.6276, OA 97.63 %, threshold 0.45).

## How to run v2

```bash
python sanity_check.py                 # 1–3 min: verifies every fix on real data (run first)
python train.py                        # main model, RGB + Canny (writes checkpoints/)
python evaluate.py                     # test set, full resolution, validation threshold
python evaluate.py --tta               # + test-time augmentation
python test_all_inference.py           # per-image results, raw vs post-processed
python app.py                          # GUI

# Ablations for the thesis (same settings, one factor changed)
python train.py --no-canny                                              # RGB only (no DSP channel)
python train.py --edge-weight 0 --out-dir checkpoints/ablation_no_edge  # no edge supervision
python evaluate.py --checkpoint checkpoints/ablation_rgb_only/best_model.pth --out-dir evaluation_results_rgb_only
```

If training is interrupted: `python train.py --resume`.

## Bugs fixed

| # | Component | v1 behaviour | v2 fix |
|---|---|---|---|
| 1 | `models/fusion.py` BAM | Dates stacked as `[B,C,H,W,2]→view(2B,…)` interleave samples, but the output was split as `out[:B]`/`out[B:]`. With batch 4, sample 1's "before" received sample 0's *after* attention. Predictions depended on batch composition (batch 4 in eval vs batch 1 in the GUI). | Dates concatenated along width; batch axis never mixed. `sanity_check.py` measures the difference. |
| 2 | BAM | Attention within each image only (spatial). | True spatial-**temporal** attention: every pixel attends to both dates (STANet). Key/value pooling makes it laptop-friendly; γ = 0 initialisation (starts as identity). |
| 3 | Input normalisation | RGB in [0, 255] next to Canny in [0, 1] → the edge channel had ≈ 0.4 % of the RGB magnitude; the DSP channel was effectively ignored. Evaluation figures were white. | RGB: ImageNet mean/std. Canny: standardised with measured statistics (edge density 0.121 → mean 0.12, std 0.33). All 4 channels ≈ zero mean, unit variance. |
| 4 | Resolution | 1024 → 256 resize (4× down-sampling); 16 × 16 px buildings became 4 × 4 px. | Training on native-resolution 256 × 256 random crops (4 per image per epoch, 30 % centred on changes). Evaluation on full 1024 × 1024 images with an overlapping (64 px), cosine-blended sliding window. |
| 5 | Decoder | Bottleneck concatenated with the *same* stage-4 tensor at 1/32 (no up-sampling, duplicated features); output at 1/4 then bilinear ×4 → blurred outlines. | Proper U-Net path 1/32 → 1/2 using a new 1/2-resolution skip (conv1 before max-pool), then ×2 + learnable full-resolution refinement. |
| 6 | Stage-0 fusion | Computed (with an expensive 4096 × 4096 BAM) and then discarded. | Used as the 1/2-resolution skip (no BAM). |
| 7 | Edge head | Saw only the **Before** encoder features, but was supervised with the edges of the change mask (new buildings exist only in After). Dead-end head. | Uses the **fused** bi-temporal features + bottleneck context, and its features are fed into the decoder (edge-guided fusion, EGCTNet-style). |
| 8 | Edge loss | `ClassBalancedEdgeLoss` existed but `train.py` used plain BCE (edges are ~1–2 % of pixels → head learns "no edge"). | Class-balanced BCE is used. |
| 9 | Validation / model selection | F1 averaged over batches of 4 (neither per-image nor dataset-level). | Dataset-level F1 on full-resolution validation images; best threshold found on validation every epoch and stored with the checkpoint. |
| 10 | Threshold | Hard-coded 0.45 in 10 files. | Selected on the validation set, saved to `checkpoints/best_threshold.json`, read everywhere (`config.load_best_threshold()`). |
| 11 | Reported IoU | 0.658 (per-image mean, inflated by empty images scored 1.0) vs dataset-level F1. | `evaluate.py` reports dataset-level metrics (IoU = F1/(2−F1)) and per-image means separately, clearly labelled. |
| 12 | GUI | Two diverging copies; model reloaded on every click; images resized to 256; colour bar added on every redraw; "Change Area" showed the GT area; GT not cropped when centre-crop was on. | One GUI (`gui/app.py`, `app.py` is a launcher); model loaded once in a thread; native resolution; TTA, post-processing and resolution options; correct predicted / true area; GT follows the same crop/scale. |
| 13 | Reproducibility | `RANDOM_SEED` never used. | Seeds for Python / NumPy / PyTorch and DataLoader workers. |
| 14 | Misc | Duplicate `dataset/dsp_processing.py`; `visualization_slide.py` used Canny 100/200 (training uses 50/150); `BatchBalancedContrastiveLoss` used D instead of D². | Single DSP module; slide figures use the config thresholds; D². |

## Other improvements

- Independent photometric augmentation per date (brightness / contrast / colour) — teaches the model that illumination and season differences are not change (relevant to the Purbachal domain shift).
- Warm-up + cosine LR schedule stepped per iteration, gradient clipping, early stopping, resumable training.
- Test-time augmentation (4 flips) in `evaluate.py` and the GUI.
- `evaluate.py`: six-panel figures with an error map (TP white, FP red, FN cyan), threshold curves, per-image CSV, Cohen's κ.
- `test_all_inference.py`: measures raw vs post-processed metrics; works on unlabelled folders (e.g. Purbachal) and supports `--scale` for imagery with a different ground resolution.

## Final v2 results (test set, 128 images, dataset-level, threshold 0.41)

| Run | F1 | IoU | Precision | Recall |
|---|---|---|---|---|
| v1 (for comparison) | 77.12 % | 62.76 % | 75.78 % | 78.50 % |
| v2, no TTA | 90.77 % | 83.10 % | 91.83 % | 89.74 % |
| v2 + TTA (final) | **91.17 %** | **83.78 %** | 91.76 % | 90.59 % |
| v2 + TTA, RGB only (`--no-canny`) | 91.63 % | 84.56 % | 91.78 % | 91.48 % |

Best validation F1 0.9107 at epoch 32 (early stop at epoch 47, ≈ 6.3 min/epoch on Apple MPS).
Batch-independence test: v1 BAM changed predictions by up to 0.241 depending on the batch; v2 gives 0.
The RGB-only ablation shows that the fixed Canny channel gives no measurable gain on LEVIR-CD.

## Notes for the thesis

- Report **dataset-level** Precision / Recall / F1 / IoU (the LEVIR-CD convention), threshold chosen on validation.
- The "empty-image" rule (F1 = IoU = 1 when TP = FP = FN = 0) only affects per-image averages; it does not affect training (metrics are not in the gradient) or dataset-level results.
- The RGB-only ablation (`--no-canny`) tests the DSP contribution; both runs are reported (no measurable gain from Canny).
- v1 vs v2 is a legitimate comparison row: same data split, same test set.
