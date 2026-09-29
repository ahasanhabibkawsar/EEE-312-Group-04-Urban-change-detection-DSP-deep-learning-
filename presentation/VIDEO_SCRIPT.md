# Video Script – Group 04, EEE 312 (Section B2)

**Project:** Digital Signal Processing and Deep Learning-Based Urban Change Detection Using Satellite Imagery  
**Target length:** about 10 minutes (1365 words at a calm 140 words per minute)
**Estimated speaking time:** 9 min 44 s plus slide changes

## How to record

1. Open `Group04_EEE312_Urban_Change_Detection.pptx`, start Slide Show, and record the screen with QuickTime (File → New Screen Recording) or Zoom.
2. The same words are in each slide's **speaker notes**, so Presenter View shows them while you record.
3. Each member records the slides marked with their name (the footer of each slide says who presents it). If one person records everything, read it in order.
4. On slide 11, switch to the running GUI for 15–20 seconds: load test_102 Before/After/GT, press Run Inference, move the threshold slider, then return to the slides.
5. Upload to YouTube with the title: **Urban Change Detection with DSP & Deep Learning | B2.04 | EEE 312 (Jan 2026) Project | Dept of EEE, BUET**, then paste the link on slide 20.

## Slide 1 – Title
*Fahim Shahriyar (2206114) · about 21 s*

Assalamu Alaikum. We are Group 04 from Section B2 of EEE 312, Digital Signal Processing One Laboratory. Our project is Digital Signal Processing and Deep Learning-Based Urban Change Detection Using Satellite Imagery. I am Fahim Shahriyar, and my teammates are Mustasin Rahman, Md. Ahasan Habib Kawsar and Md. Kawsar Ahmed.

## Slide 2 – Outline
*Fahim Shahriyar (2206114) · about 13 s*

We will first summarise the project and the problem, then explain our design, show the working software, evaluate it against the program outcomes, and finish with teamwork, future work and references.

## Slide 3 – 1. Summary / Abstract
*Fahim Shahriyar (2206114) · about 36 s*

Our system takes two satellite images of the same place, taken years apart, and marks every pixel where a new building has appeared. A Gaussian filter and the Canny edge detector extract structural edges, which we add to the colour image as a fourth channel. A Siamese neural network with attention then compares the two dates. On 128 unseen test images of the LEVIR-CD dataset, it reaches an F1-score of 91.2 percent and an IoU of 83.8 percent, and it runs in a desktop application.

## Slide 4 – 2. Introduction
*Fahim Shahriyar (2206114) · about 34 s*

Cities like Dhaka grow faster than manual surveys can map them. Planners need to know where new buildings appear: unauthorised construction, encroachment on wetlands, or damage after a disaster. The task is hard for three reasons: shadows, seasons and sensor colour change the image even when nothing is built; only about 4.6 percent of pixels actually change; and a house is only ten to twenty pixels wide. On the right, an empty field has become a row of warehouses.

## Slide 5 – 3.1 Design: Literature Review
*Fahim Shahriyar (2206114) · about 33 s*

Early methods simply subtracted the two images, which is very sensitive to lighting. Siamese networks pass both images through the same encoder; STANet added attention and published LEVIR-CD, and transformers such as BIT and ChangeFormer pushed F1 above 89 percent. Our aim was a lighter design that trains on a laptop and adds an explicit DSP edge channel. The chart compares our test F1 with published results; our inference settings differ slightly, so it is an indicative comparison.

## Slide 6 – 3.2 Design Methods (PO(a))
*Mustasin Rahman (2206104) · about 30 s*

I am Mustasin Rahman, and I will explain the design. This is the complete pipeline. The before and after images first pass through our DSP front-end, shown in red: grayscale conversion, Gaussian smoothing and Canny edge detection. The edge map is stacked with the colour channels and standardised. The dark blocks are the deep-learning stages: a shared ResNet-34 encoder, attention-based fusion, and a U-Net decoder that produces the change map.

## Slide 7 – 3.2 Design Methods: DSP Front-End
*Mustasin Rahman (2206104) · about 36 s*

Here is each DSP stage on a real image. We convolve the grayscale image with a five-by-five Gaussian kernel, sigma equal to one; this low-pass filter removes noise that would create false edges. Canny then computes Sobel gradients, thins edges by non-maximum suppression, and applies hysteresis thresholds of 50 and 150. Finally, every channel is standardised to zero mean and unit variance. This was critical: in our first version the edge channel was 255 times weaker than the colour channels, so the network ignored it.

## Slide 8 – 3.2 Design Methods: Network Architecture
*Md. Ahasan Habib Kawsar (2206119) · about 35 s*

I am Md. Ahasan Habib Kawsar, and I will explain the network. Both four-channel images pass through the same pre-trained ResNet-34 encoder, giving features at five scales, and at every scale a fusion block compares the two dates. The decoder rebuilds a full-resolution change map. An edge head predicts the outlines of changed buildings and feeds them into the decoder, which sharpens boundaries. Large images are predicted as overlapping 256-pixel tiles, so the network always works at the original half-metre resolution.

## Slide 9 – 3.2 Design Methods: Attention, Fusion, Loss
*Md. Ahasan Habib Kawsar (2206119) · about 27 s*

The attention module places both dates side by side, so every pixel can attend to every pixel of both images. The fusion block keeps the before features, the after features and their absolute difference. The loss adds binary cross-entropy, Dice loss for the class imbalance, and a class-balanced edge loss. We evaluate with precision, recall, F1 and IoU over all test pixels.

## Slide 10 – 3.3 Design: Training and Inference
*Md. Ahasan Habib Kawsar (2206119) · about 26 s*

We train on random 256-pixel crops of the full-resolution images, so small buildings keep their detail. Both dates are flipped and rotated together, while brightness and colour are changed separately, so lighting differences are not learned as change. Training ran on a MacBook GPU; validation F1 reached 0.91 at epoch 32. The threshold, 0.41, was chosen on the validation set only.

## Slide 11 – 4 Implementation: Demonstration
*Md. Kawsar Ahmed (2206122) · about 44 s*

I am Md. Kawsar Ahmed, and I will show the software. This is our PyQt5 desktop application. We load the before image, the after image and, optionally, the ground truth, and press Run Inference. The model runs in a background thread, so the window never freezes. The bottom row shows the change probability, the final mask, and the detected buildings in red on the after image. When I move the threshold slider, the mask and the metrics update instantly. For this test image, F1 is 95.8 percent and IoU is 91.9 percent. Post-processing removes small noise and straightens building outlines into polygons.

## Slide 12 – 4.1 Implementation: Results Gallery
*Md. Kawsar Ahmed (2206122) · about 24 s*

Here are three more test results. From left to right: before, after, ground truth, probability, prediction, and an error map, where white is correct, red is a false alarm and cyan is a miss. The outlines are sharp, and even closely spaced houses in a new housing estate are separated, with F1 between 0.92 and 0.98.

## Slide 13 – 5. Analysis and Evaluation: Test Results
*Md. Ahasan Habib Kawsar (2206119) · about 25 s*

On all 128 test images, the final model reaches 91.8 percent precision, 90.6 percent recall, 91.2 percent F1 and 83.8 percent IoU; that is 14 F1 points and 21 IoU points above our first version. False alarms and misses are balanced, and the validation threshold is almost exactly the best test threshold, so the result is not over-tuned.

## Slide 14 – 5.1 Novelty
*Md. Ahasan Habib Kawsar (2206119) · about 24 s*

Our design is novel in four ways: a standardised DSP edge channel fused with colour inside a pre-trained encoder; attention and difference-based fusion at every scale; an edge-guided decoder supervised with a Sobel operator; and native-resolution deployment in a real-time application. Finding and fixing three silent bugs in our first version added 14 F1 points.

## Slide 15 – 5.2 Design Considerations (PO(c))
*Mustasin Rahman (2206104) · about 36 s*

For public health and safety, the system can flag unauthorised construction in flood-prone or earthquake-risk areas, and map damage quickly after a disaster. Because a missed building could hide a hazard, the output is decision support, and a human always verifies it. Environmentally, it monitors encroachment on wetlands and farmland without field trips. For society, it can support city planners, and the threshold slider lets users choose between fewer false alarms and fewer misses. It must not be used to target informal settlements.

## Slide 16 – 5.3 Investigations (PO(d))
*Md. Ahasan Habib Kawsar (2206119) · about 26 s*

A controlled test showed that our old attention module changed an image's output by 0.24 depending on the other images in its batch; now the difference is zero. Theory predicts IoU equals F1 over two minus F1, which gives 0.8377; we measured 0.8378. In our worst test image, tiny mobile homes are missed and re-roofed buildings are mistaken for new ones.

## Slide 17 – 5.4 Limitations of Tools (PO(e))
*Mustasin Rahman (2206104) · about 30 s*

Our tools have limits. Training on a laptop GPU restricted the batch size to four and took about five hours. LEVIR-CD contains only buildings in American cities at half-metre resolution, so dense Bangladeshi areas are a domain shift. Canny thresholds are fixed, objects smaller than about two metres are unreliable, and badly aligned image pairs cause false alarms. We reduced these effects with native-resolution tiling, test-time augmentation and a validation-selected threshold.

## Slide 18 – 5.5 Ethical Issues (PO(h))
*Mustasin Rahman (2206104) · about 26 s*

On ethics: we used the public LEVIR-CD dataset under its licence and cite every source. High-resolution imagery can reveal private property, so the intended use is planning and disaster response, not surveillance. We never used the test set for tuning, and we report our earlier bugs and failure cases openly. Our use of AI coding assistance is disclosed in the report.

## Slide 19 – 6.1 Individual Contribution (PO(i))
*Md. Kawsar Ahmed (2206122) · about 23 s*

Each member led one part. Mustasin prepared the data and the DSP pre-processing, Fahim led the literature review and the error analysis, Ahasan designed and trained the network, and I built the GUI and the inference pipeline. We met weekly, shared one code folder, and every module was reviewed by a second member.

## Slide 20 – 7 Communication to External Stakeholders (PO(j))
*Md. Kawsar Ahmed (2206122) · about 11 s*

Our code, the trained model and a user manual are available on GitHub, and this video is on YouTube. The links are on this slide.

## Slide 21 – 8. Future Work (PO(l))
*Md. Kawsar Ahmed (2206122) · about 21 s*

First, we will finish the RGB-only comparison to measure exactly how much the edge channel adds. We also want to label image pairs from Dhaka and Purbachal, detect roads and demolition as separate classes, try transformer backbones and longer image time series, and turn the tool into a GIS plug-in.

## Slide 22 – 9. References
*Md. Kawsar Ahmed (2206122) · about 3 s*

These are our references. Thank you for watching.

## Viva preparation – likely questions

**Q: Does the Canny channel actually improve the result?**  
A: We made the edge channel equally strong as the colour channels, which the first version did not do. A full RGB-only comparison run was started, but it did not finish before the deadline, so we do not claim a measured gain; that comparison is listed as future work.

**Q: Why F1 and IoU instead of accuracy?**  
A: Only about 5 percent of pixels change, so a model that predicts 'no change' everywhere would already get about 95 percent accuracy. F1 and IoU measure the changed class.

**Q: Why is IoU lower than F1?**  
A: For the same counts, IoU = F1 / (2 − F1), so it is always lower; 0.9117 gives 0.8377, which matches our measurement.

**Q: How did you choose the threshold 0.41?**  
A: By sweeping thresholds on the 64 validation images and picking the best F1. The test set was never used for that choice.

**Q: Why tiles instead of resizing the 1024 × 1024 image?**  
A: Resizing to 256 shrinks a 16-pixel house to 4 pixels. Tiles keep the native 0.5 m/pixel detail; overlapping tiles are blended to avoid seams.

**Q: What was wrong in version 1?**  
A: The attention module mixed samples within a batch, the edge channel was 255 times weaker than RGB, and images were down-sampled 4×. Fixing these raised F1 from 0.77 to 0.91.
