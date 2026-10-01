# Video Script – Group 04, EEE 312 (Section B2)

**Project:** Digital Signal Processing and Deep Learning-Based Urban Change Detection Using Satellite Imagery  
**Length:** 1301 words ≈ 9 min 18 s at 140 words per minute, plus about 30 s of GUI demo – inside the 10-minute limit

## Before recording (once)

1. `source .venv/bin/activate`, then `python app.py`. Wait until the status bar says the model is ready.
2. Take a new GUI screenshot showing the **Model** menu and replace the picture on slide 16 (right-click → Change Picture). The current picture is the older GUI.
3. Keep two image pairs ready in Finder: `data/LEVIR-CD/test/{A,B,label}/test_102.png` and `data/LEVIR-CD_purbachal/test/{A,B}/test_2.png`.
4. Close other apps, turn on Do Not Disturb, and use a headset microphone if you have one. Set the screen to 1920 × 1080 if possible.

## How to record

1. Open `Group04_EEE312_Final_Presentation.pptx` → Slide Show → Presenter View. The exact words for every slide are in the speaker notes and below.
2. Record with QuickTime (File → New Screen Recording, choose the microphone under Options) or a Zoom meeting with all four members, recorded to the computer.
3. Each member speaks on the slides whose footer shows their ID. Speak a little slower than normal; pause one second after each slide change.
4. **GUI demo (slide 16, about 40 s, while reading the slide 16 words):**
   - Load Before / After / Ground truth of `test_102`, keep Model = *v2 – RGB + Canny (final)*, press **Run Inference**, move the threshold slider once.
   - Load the Purbachal pair `test_2` (no ground truth) and press **Compare all models**; leave the comparison window on screen for 5 s.
   - Switch back to the slides.
5. Slide 18 explains the Purbachal result you just showed; slide 21 gives the measured comparison.
6. Export as 1080p MP4 and check: every voice is audible, no notifications, total under 10 minutes.
7. Upload to YouTube (Visibility: Unlisted or Public) with the title **Urban Change Detection with DSP & Deep Learning | B2.04 | EEE 312 (Jan 2026) Project | Dept of EEE, BUET**.
8. Put the link on slide 26, in section 7.4 of the report and in the README.

| Presenter | Slides | Time |
|---|---|---|
| Md. Ahasan Habib Kawsar (2206119) | 1, 2, 3, 4, 6, 13, 14, 15, 21 | ≈ 3 min 07 s |
| Mustasin Rahman (2206104) | 7, 8, 9, 10, 11, 12, 23, 24 | ≈ 2 min 37 s |
| Fahim Shahriyar (2206114) | 5, 17, 18, 19, 20, 22 | ≈ 2 min 14 s |
| Md. Kawsar Ahmed (2206122) | 16, 25, 26, 27, 28 | ≈ 1 min 19 s |

## Slide 1 – Title
*Md. Ahasan Habib Kawsar (2206119) · about 21 s*

Assalamu Alaikum. We are Group 04 from Section B2 of EEE 312, Digital Signal Processing One Laboratory. Our project is Digital Signal Processing and Deep Learning-Based Urban Change Detection Using Satellite Imagery. I am Md. Ahasan Habib Kawsar, and my teammates are Mustasin Rahman, Fahim Shahriyar and Md. Kawsar Ahmed.

## Slide 2 – Outline
*Md. Ahasan Habib Kawsar (2206119) · about 12 s*

We will summarise the project, explain the problem and our design, demonstrate the software, evaluate the results against the program outcomes, and finish with teamwork, future work and references.

## Slide 3 – 1. Abstract
*Md. Ahasan Habib Kawsar (2206119) · about 25 s*

Our system takes two satellite images of the same place, taken years apart, and marks every pixel where a building has appeared. A Gaussian filter and the Canny detector extract edges as a fourth input channel, and a Siamese network compares the two dates. On 128 unseen test images it reaches 91.2 percent F1 and 83.8 percent IoU.

## Slide 4 – 2. Introduction
*Md. Ahasan Habib Kawsar (2206119) · about 26 s*

Cities like Dhaka grow faster than manual surveys can map them, so planners need an automatic way to find new construction and post-disaster damage. The task is hard for three reasons: shadows and seasons change the image even when nothing is built; only a small fraction of pixels actually change; and a house is only ten to twenty pixels wide.

## Slide 5 – 3.1 Literature Review
*Fahim Shahriyar (2206114) · about 29 s*

Classical methods simply subtract the two images, which is fast but very sensitive to lighting. Siamese networks pass both images through the same encoder, and attention and transformer models such as STANet, BIT and ChangeFormer reach around ninety percent F1. We combine explicit DSP edge features with attention in a network that trains on a laptop. The chart compares our F1 with published results; the protocols differ slightly.

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
*Md. Kawsar Ahmed (2206122) · about 33 s*

I am Md. Kawsar Ahmed, and this is our PyQt5 application. We load the before and after images and the ground truth, and press Run Inference; the model runs in a background thread, so the window never freezes. The Model menu lets us switch between the final model, the RGB-only model, our old model and two ensembles, and Compare all models shows them side by side. On this test image, the final model reaches 95.8 percent F1.

## Slide 17 – 4.1 Results on LEVIR-CD
*Fahim Shahriyar (2206114) · about 19 s*

I am Fahim Shahriyar. Here an empty field has become a new housing estate. The probability map is confident on the new houses and clean on the background, and the error map shows mostly white, meaning correct pixels. F1 on this image is 0.939.

## Slide 18 – 4.2 Bangladeshi Imagery: Old vs New
*Fahim Shahriyar (2206114) · about 30 s*

We also tested twelve image pairs from Purbachal in Dhaka, where open plots became dense apartment blocks. Here our old model marks almost ten percent of the pixels as changed, while the final model marks under two percent, so the old model seems to catch more of the new construction. But these images have no labels, so we cannot score them, and making the input larger did not help.

## Slide 19 – 5.1 Quantitative Results
*Fahim Shahriyar (2206114) · about 20 s*

On all 128 test images the final model reaches 91.8 percent precision, 90.6 percent recall, 91.2 percent F1 and 83.8 percent IoU. False alarms and misses are balanced. F1 rises with the amount of change, from 84 percent for small changes to 92 percent for large ones.

## Slide 20 – 5.2 Investigations (PO(d))
*Fahim Shahriyar (2206114) · about 21 s*

We ran controlled experiments. Fixing three bugs in our first version raised F1 from 77 to 91 percent, and test-time augmentation added 0.4 points. Our key finding: without the Canny channel, F1 did not drop; it was 91.6 percent with RGB only, because the encoder already learns edge filters itself.

## Slide 21 – 5.3 Old vs New Model, 3 vs 4 Channels
*Md. Ahasan Habib Kawsar (2206119) · about 29 s*

To compare the old and new models fairly, we also tested on WHU-CD, a labelled dataset from New Zealand that neither model has seen. The final model reaches 82.9 percent F1 there, and the old model only 66.7 percent, with many false alarms. The RGB-only model is as good as the four-channel model on both datasets, and combining the old and new models gave no real gain.

## Slide 22 – 5.4 Limitations & Failure Modes (PO(e))
*Fahim Shahriyar (2206114) · about 15 s*

Tiny mobile homes, only three to five pixels wide, are missed; re-roofed buildings are mistaken for new ones; and dense Bangladeshi scenes cause a domain shift. A laptop GPU allowed only batch size four.

## Slide 23 – 5.5 Design Considerations (PO(c))
*Mustasin Rahman (2206104) · about 17 s*

For public health and safety, the system can flag unauthorised construction in flood-prone or earthquake-risk areas and map damage after a disaster, but a human always verifies the output. It also monitors encroachment on wetlands and farmland without field trips.

## Slide 24 – 5.6 Ethical Issues (PO(h))
*Mustasin Rahman (2206104) · about 21 s*

We used the public LEVIR-CD dataset under its licence and cite every source. High-resolution imagery can reveal private property, so the intended use is planning and disaster response, not surveillance or eviction. We never tuned on the test set, and we report our negative DSP result and failure cases openly.

## Slide 25 – 6. Individual Contribution (PO(i))
*Md. Kawsar Ahmed (2206122) · about 17 s*

Each member led one part: Mustasin the data and DSP pre-processing, Fahim the literature review and error analysis, Ahasan the network and training, and I built the GUI and inference pipeline. Every module was reviewed by a second member.

## Slide 26 – 7. Communication (PO(j))
*Md. Kawsar Ahmed (2206122) · about 9 s*

Our code, trained model and user manual are on GitHub, and this video is on YouTube. The links are on this slide.

## Slide 27 – 8. Future Work (PO(l))
*Md. Kawsar Ahmed (2206122) · about 17 s*

Next, we want to build a labelled dataset of Dhaka image pairs and fine-tune the model on it, learn adaptive edge filters instead of fixed Canny thresholds, classify the type of change, and deploy the tool for large-scale imagery.

## Slide 28 – 9. References
*Md. Kawsar Ahmed (2206122) · about 3 s*

These are our references. Thank you for watching.

## Viva preparation – likely questions

**Q: Your old model detects more change on the Dhaka images. Why not use it?**  
A: Those images have no ground truth, so we cannot measure who is right. On labelled data the old model is clearly worse: 78.2 % vs 91.2 % F1 on LEVIR-CD and 66.7 % vs 82.9 % on WHU-CD, an unseen city, where its precision is only 57 % (many false alarms). It marks more pixels everywhere, not only correct ones. Both models are kept in the GUI, so a user can compare them.

**Q: Why does the old model react more strongly to Purbachal?**  
A: It resizes every image to 256 × 256 and uses unnormalised RGB, so it reacts to large brightness and texture differences. Purbachal changed from open plots to dense blocks, which is a huge appearance change. The final model was trained with independent brightness/colour jitter per date to ignore such differences, which makes it conservative on unfamiliar scenes.

**Q: Did combining the two models help?**  
A: No real gain. Averaging scored 89.1 % on LEVIR-CD (−2.0) and 83.0 % on WHU-CD (+0.2); the union (max) added false alarms (87.2 % and 76.8 %).

**Q: Does the Canny channel actually improve the result?**  
A: No measurable gain. RGB-only scored 91.7 % vs 91.2 % on LEVIR-CD and 83.1 % vs 82.9 % on WHU-CD. On Purbachal the RGB-only model even marks more change (7.9 % vs 1.8 % of pixels). The ImageNet-pre-trained first layer already learns edge filters. DSP still matters elsewhere: Gaussian/Canny preprocessing, Sobel edge supervision, cosine-window tile blending and morphological post-processing.

**Q: Is the problem on Dhaka images the resolution?**  
A: We tested it: enlarging the input 2× or 4× gave even fewer detections (0.44 % and 0.01 %). On WHU-CD, matching the 0.3 m/px resolution helped (+1.8 F1), but for Purbachal the issue is domain shift: dense high-rise blocks, shadows and local roofs that LEVIR-CD's US suburbs do not contain.

**Q: How would you fix it?**  
A: Label a set of Purbachal / Dhaka pairs and fine-tune the final model on them, then measure F1 on held-out local pairs.

**Q: Why F1 and IoU instead of accuracy?**  
A: Only about 5 % of pixels change, so predicting 'no change' everywhere already gives about 95 % accuracy. F1 and IoU measure the changed class.

**Q: How was the threshold 0.41 chosen?**  
A: By sweeping thresholds on the 64 validation images. The test set was never used for any choice; the same rule picked the thresholds for the old model and the ensembles.

**Q: Why tiles instead of resizing the 1024 × 1024 image?**  
A: Resizing to 256 shrinks a 16-pixel house to 4 pixels. Overlapping 256-px tiles keep the native 0.5 m/pixel detail and are blended with a cosine window to avoid seams.

**Q: What was wrong in version 1?**  
A: The attention module mixed samples within a batch, the edge channel was 255 times weaker than RGB, and images were down-sampled 4×. Fixing these raised F1 from 77.1 % to 91.2 %.
