# Video Script – Group 04, EEE 312 (Section B2)

**Project:** Digital Signal Processing and Deep Learning-Based Urban Change Detection Using Satellite Imagery  
**Length:** 1295 words ≈ 9 min 14 s at 140 words per minute (plus slide changes) – within the 10-minute limit

## How to record

1. Open `Group04_EEE312_Final_Presentation.pptx` → Slide Show → Presenter View. The exact words for every slide are in the speaker notes.
2. Record the screen with QuickTime (File → New Screen Recording) or Zoom. Each member speaks on the slides whose footer shows their ID.
3. On slide 16, switch to the running GUI for 15–20 seconds: load test_102 Before / After / Ground truth, press Run Inference, move the threshold slider, then return to the slides.
4. Upload to YouTube as: **Urban Change Detection with DSP & Deep Learning | B2.04 | EEE 312 (Jan 2026) Project | Dept of EEE, BUET**, then put the link on slide 25.

| Presenter | Slides | Time |
|---|---|---|
| Md. Ahasan Habib Kawsar (2206119) | 1, 2, 3, 4, 6, 13, 14, 15 | ≈ 2 min 45 s |
| Mustasin Rahman (2206104) | 7, 8, 9, 10, 11, 12, 22, 23 | ≈ 2 min 41 s |
| Fahim Shahriyar (2206114) | 5, 17, 18, 19, 20, 21 | ≈ 2 min 35 s |
| Md. Kawsar Ahmed (2206122) | 16, 24, 25, 26, 27 | ≈ 1 min 13 s |

## Slide 1 – Title
*Md. Ahasan Habib Kawsar (2206119) · about 21 s*

Assalamu Alaikum. We are Group 04 from Section B2 of EEE 312, Digital Signal Processing One Laboratory. Our project is Digital Signal Processing and Deep Learning-Based Urban Change Detection Using Satellite Imagery. I am Md. Ahasan Habib Kawsar, and my teammates are Mustasin Rahman, Fahim Shahriyar and Md. Kawsar Ahmed.

## Slide 2 – Outline
*Md. Ahasan Habib Kawsar (2206119) · about 12 s*

We will summarise the project, explain the problem and our design, demonstrate the software, evaluate the results against the program outcomes, and finish with teamwork, future work and references.

## Slide 3 – 1. Abstract
*Md. Ahasan Habib Kawsar (2206119) · about 31 s*

Our system takes two satellite images of the same place, taken years apart, and marks every pixel where a building has appeared. A Gaussian filter and the Canny edge detector extract structural edges, which we add to the colour image as a fourth channel, and a Siamese neural network compares the two dates. On 128 unseen LEVIR-CD test images it reaches an F1-score of 91.2 percent and an IoU of 83.8 percent.

## Slide 4 – 2. Introduction
*Md. Ahasan Habib Kawsar (2206119) · about 26 s*

Cities like Dhaka grow faster than manual surveys can map them, so planners need an automatic way to find new construction and post-disaster damage. The task is hard for three reasons: shadows and seasons change the image even when nothing is built; only a small fraction of pixels actually change; and a house is only ten to twenty pixels wide.

## Slide 5 – 3.1 Literature Review
*Fahim Shahriyar (2206114) · about 33 s*

Classical methods simply subtract the two images, which is fast but very sensitive to lighting. Siamese networks pass both images through the same encoder, and attention and transformer models such as STANet, BIT and ChangeFormer reach around ninety percent F1. Our approach combines explicit DSP edge features with attention in a lightweight network that trains on a laptop. The chart compares our test F1 with published results; the protocols differ slightly, so it is an indicative comparison.

## Slide 6 – 3.2 Design Methods (PO(a))
*Md. Ahasan Habib Kawsar (2206119) · about 14 s*

This is our complete design flow in six steps: data preparation, preprocessing, DSP feature engineering, the Siamese U-Net model, training, and finally evaluation and demonstration. My teammates will now explain each step.

## Slide 7 – 3.2.1 Data Preparation
*Mustasin Rahman (2206104) · about 22 s*

I am Mustasin Rahman. We use the LEVIR-CD dataset: 637 pairs of before and after images, each 1024 by 1024 pixels at half a metre per pixel, with a ground-truth mask that marks every changed pixel. The official split is 445 pairs for training, 64 for validation and 128 for testing.

## Slide 8 – 3.2.2 Image Preprocessing
*Mustasin Rahman (2206104) · about 27 s*

Instead of shrinking the images, we cut random 256 by 256 crops at full resolution, so small buildings keep their detail. Flips and rotations are applied identically to both dates and the mask, while brightness and colour are changed separately for each date. Every channel is then standardised. At test time, the full image is covered by 25 overlapping tiles that are blended smoothly.

## Slide 9 – 3.2.3 DSP Feature Engineering
*Mustasin Rahman (2206104) · about 17 s*

Our DSP front-end turns each RGB image into a four-channel input. The image is converted to grayscale, smoothed with a Gaussian filter, and passed through the Canny edge detector; the edge map is then stacked with the original colour channels.

## Slide 10 – Grayscale Conversion
*Mustasin Rahman (2206104) · about 15 s*

Grayscale conversion uses the ITU-R BT.601 weights: Y equals 0.299 R plus 0.587 G plus 0.114 B. This reduces three colour channels to one intensity map while keeping the geometry of buildings for gradient analysis.

## Slide 11 – Gaussian Filtering
*Mustasin Rahman (2206104) · about 15 s*

Next, we convolve the grayscale image with a five-by-five Gaussian kernel with sigma equal to one. This low-pass filter removes high-frequency noise, so the gradient operators respond to real structural edges instead of pixel noise.

## Slide 12 – Canny Edge Detection & Fusion
*Mustasin Rahman (2206104) · about 22 s*

Canny computes the horizontal and vertical Sobel gradients and their magnitude, thins the edges by non-maximum suppression, and applies hysteresis thresholds of 50 and 150. The binary edge map is appended as a fourth channel, and all four channels are standardised so that the edge channel carries the same weight as colour.

## Slide 13 – 3.2.4 Siamese U-Net Architecture
*Md. Ahasan Habib Kawsar (2206119) · about 20 s*

Both four-channel images pass through the same pre-trained ResNet-34 encoder, which gives features at five scales. At every scale, a fusion block compares the two dates. The U-Net decoder rebuilds a full-resolution change map, guided by an edge head that predicts the outlines of changed buildings.

## Slide 14 – 3.2.5 Attention, Fusion, Loss
*Md. Ahasan Habib Kawsar (2206119) · about 21 s*

The attention module places both dates side by side, so every pixel can attend to every pixel of both images. The fusion keeps the before features, the after features and their absolute difference. The loss combines binary cross-entropy, Dice loss for the class imbalance, and a class-balanced edge loss.

## Slide 15 – 3.3 Training and Inference
*Md. Ahasan Habib Kawsar (2206119) · about 20 s*

We trained with the AdamW optimiser and a cosine learning-rate schedule on a MacBook GPU, at about six minutes per epoch. Validation F1 reached 0.91 at epoch 32, and training stopped automatically at epoch 47. The decision threshold, 0.41, was chosen on the validation set only.

## Slide 16 – 4. Implementation: Demonstration
*Md. Kawsar Ahmed (2206122) · about 27 s*

I am Md. Kawsar Ahmed, and this is our PyQt5 desktop application. We load the before and after images and, optionally, the ground truth, and press Run Inference. The model runs in a background thread, so the window never freezes. Moving the threshold slider updates the mask and the metrics instantly. On this test image, F1 is 95.8 percent and IoU is 91.9 percent.

## Slide 17 – 4.1 Results on LEVIR-CD
*Fahim Shahriyar (2206114) · about 19 s*

I am Fahim Shahriyar. Here an empty field has become a new housing estate. The probability map is confident on the new houses and clean on the background, and the error map shows mostly white, meaning correct pixels. F1 on this image is 0.939.

## Slide 18 – 4.2 Inference on Dhaka Imagery
*Fahim Shahriyar (2206114) · about 21 s*

We also tested real Bangladeshi imagery from Bashundhara. Here the model detects only a few changes near construction sites and misses much of the real change. Dense layouts, high-rise shadows and local building styles are very different from the American suburbs in LEVIR-CD, so local training data is needed.

## Slide 19 – 5.1 Quantitative Results
*Fahim Shahriyar (2206114) · about 27 s*

On all 128 test images the final model reaches 91.8 percent precision, 90.6 percent recall, 91.2 percent F1 and 83.8 percent IoU. False alarms and misses are balanced, at about half a percent of pixels each. The right chart shows that F1 rises with the amount of change in an image, from 84 percent for small changes to 92 percent for large developments.

## Slide 20 – 5.2 Investigations (PO(d))
*Fahim Shahriyar (2206114) · about 33 s*

We ran controlled experiments. Fixing three bugs in our first version raised F1 from 77 to 91 percent, and test-time augmentation added 0.4 points. Our most important finding: when we removed the Canny channel, F1 did not drop — it was 91.6 percent with RGB only. So on this dataset the fixed edge channel gives no measurable gain, because the encoder already learns edge filters itself. The measured IoU also matches the theoretical relation with F1 exactly.

## Slide 21 – 5.3 Limitations & Failure Modes (PO(e))
*Fahim Shahriyar (2206114) · about 22 s*

The remaining errors have clear causes. Tiny mobile homes, only three to five pixels wide, are missed; re-roofed buildings are mistaken for new ones; and dense Bangladeshi scenes cause a domain shift. Our tools also limited us: a laptop GPU allowed only batch size four, and the Canny thresholds are fixed.

## Slide 22 – 5.4 Design Considerations (PO(c))
*Mustasin Rahman (2206104) · about 21 s*

For public health and safety, the system can flag unauthorised construction in flood-prone or earthquake-risk areas and map damage after a disaster, but a human always verifies the output. It monitors encroachment on wetlands and farmland without field trips, and the threshold slider lets users balance false alarms against misses.

## Slide 23 – 5.5 Ethical Issues (PO(h))
*Mustasin Rahman (2206104) · about 22 s*

We used the public LEVIR-CD dataset under its licence and cite every source. High-resolution imagery can reveal private property, so the intended use is planning and disaster response, not surveillance or eviction. We never used the test set for tuning, and we report our negative DSP result and failure cases openly.

## Slide 24 – 6. Individual Contribution (PO(i))
*Md. Kawsar Ahmed (2206122) · about 17 s*

Each member led one part: Mustasin the data and DSP pre-processing, Fahim the literature review and error analysis, Ahasan the network and training, and I built the GUI and inference pipeline. Every module was reviewed by a second member.

## Slide 25 – 7. Communication (PO(j))
*Md. Kawsar Ahmed (2206122) · about 9 s*

Our code, trained model and user manual are on GitHub, and this video is on YouTube. The links are on this slide.

## Slide 26 – 8. Future Work (PO(l))
*Md. Kawsar Ahmed (2206122) · about 17 s*

Next, we want to build a labelled dataset of Dhaka image pairs and fine-tune the model on it, learn adaptive edge filters instead of fixed Canny thresholds, classify the type of change, and deploy the tool for large-scale imagery.

## Slide 27 – 9. References
*Md. Kawsar Ahmed (2206122) · about 3 s*

These are our references. Thank you for watching.

## Viva preparation – likely questions

**Q: Does the Canny channel actually improve the result?**  
A: No measurable gain on LEVIR-CD: RGB-only reached 91.63 % F1 versus 91.17 % with Canny, under the same protocol. The encoder's first layer, pre-trained on ImageNet, already learns edge detectors, so a fixed Canny map adds little. The DSP stages still matter elsewhere: Gaussian/Canny preprocessing, Sobel-based edge supervision, cosine-window tile blending and morphological post-processing.

**Q: Then why keep the DSP channel?**  
A: It is the method we designed and tested; reporting the negative ablation is part of an honest investigation (PO(d)). Future work replaces fixed thresholds with learnable edge filters.

**Q: Why F1 and IoU instead of accuracy?**  
A: Only about 5 % of pixels change, so predicting 'no change' everywhere already gives about 95 % accuracy. F1 and IoU measure the changed class.

**Q: Why is IoU lower than F1?**  
A: For the same counts IoU = F1 / (2 − F1) exactly; F1 = 0.9117 gives 0.8378, identical to our measured IoU.

**Q: How was the threshold 0.41 chosen?**  
A: By sweeping thresholds on the 64 validation images. The test set was never used for any choice.

**Q: Why tiles instead of resizing the 1024 × 1024 image?**  
A: Resizing to 256 shrinks a 16-pixel house to 4 pixels. Overlapping 256-px tiles keep the native 0.5 m/pixel detail and are blended with a cosine window to avoid seams.

**Q: What was wrong in version 1?**  
A: The attention module mixed samples within a batch, the edge channel was 255 times weaker than RGB, and images were down-sampled 4×. Fixing these raised F1 from 77.1 % to 91.2 %.

**Q: Why does it fail on Dhaka imagery?**  
A: Domain shift: LEVIR-CD contains low-density US suburbs; Dhaka has dense, high-rise buildings with long shadows. Local labelled data and fine-tuning are needed.
