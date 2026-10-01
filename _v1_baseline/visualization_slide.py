import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Point this to any good 'Before' or 'After' image in your dataset
INPUT_IMAGE = "dataset_output_2025_3/After/0_234.tif" 
OUTPUT_DIR = "dsp_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_dsp_visuals():
    # 1. Load Original RGB
    img_bgr = cv2.imread(INPUT_IMAGE)
    if img_bgr is None:
        print(f"Error: Could not load {INPUT_IMAGE}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{OUTPUT_DIR}/0_original.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # 2. Step 1: Grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(f"{OUTPUT_DIR}/1_grayscale.png", gray)

    # 3. Step 2: Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    cv2.imwrite(f"{OUTPUT_DIR}/2_blurred.png", blurred)

    # 4. Step 3: Canny Edge Detection
    # You can tweak 100, 200 to match your actual dsp_processing.py thresholds
    canny = cv2.Canny(blurred, 100, 200) 
    cv2.imwrite(f"{OUTPUT_DIR}/3_canny.png", canny)

    # 5. Step 4: Channel Fusion (Visualization)
    # Since a 4-channel image can't be rendered on a normal screen, 
    # we visualize the "fusion" by overlaying the Canny edges in bright Neon Green on the RGB image.
    fused_vis = img_rgb.copy()
    fused_vis[canny > 0] = [0, 255, 0] # Color the edges pure green
    cv2.imwrite(f"{OUTPUT_DIR}/4_fused_visualization.png", cv2.cvtColor(fused_vis, cv2.COLOR_RGB2BGR))

    print(f"Success! Visualizations saved to the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    generate_dsp_visuals()