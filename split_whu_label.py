import os
from PIL import Image, ImageEnhance, ImageFilter

# IMPORTANT: Prevents the DecompressionBombError for large files
Image.MAX_IMAGE_PIXELS = None

# Your exact enhancement function
def enhance_crop(img):
    """Applies the exact sharpening and contrast from your crop_dataset.py"""
    # Gentle sharpening
    img = img.filter(
        ImageFilter.UnsharpMask(
            radius=1.0,
            percent=100,
            threshold=3
        )
    )
    # Very slight contrast enhancement
    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(1.03)
    return img

# Paths
WHU_ROOT = "/Users/ahasanhabibkawsar/Documents/Urban_Change_Detection/data/WHU-CD"
BASE = os.path.join(WHU_ROOT, "The_two_period_image_data")
TILE_SIZE = 512

def process_split(split_name):
    # Exact path structure you provided
    a_dir = os.path.join(BASE, '2012', 'whole_image', split_name, 'image')
    b_dir = os.path.join(BASE, '2016', 'whole_image', split_name, 'image')
    label_dir = os.path.join(BASE, 'change_label', split_name)

    # Find the .tif files dynamically
    a_file = [f for f in os.listdir(a_dir) if f.lower().endswith('.tif')][0]
    b_file = [f for f in os.listdir(b_dir) if f.lower().endswith('.tif')][0]
    label_file = [f for f in os.listdir(label_dir) if f.lower().endswith('.tif')][0]

    print(f"Opening {split_name} images... (Memory-safe method)")
    
    # Open the huge images WITHOUT converting to numpy array (prevents crash)
    whole_a = Image.open(os.path.join(a_dir, a_file)).convert('RGB')
    whole_b = Image.open(os.path.join(b_dir, b_file)).convert('RGB')
    whole_label = Image.open(os.path.join(label_dir, label_file)).convert('L')

    w, h = whole_a.size
    print(f"Whole image size: {w} x {h}")

    # Output folders
    out_a = os.path.join(WHU_ROOT, split_name, 'A')
    out_b = os.path.join(WHU_ROOT, split_name, 'B')
    out_label = os.path.join(WHU_ROOT, split_name, 'label')
    os.makedirs(out_a, exist_ok=True)
    os.makedirs(out_b, exist_ok=True)
    os.makedirs(out_label, exist_ok=True)

    count = 0
    # Sliding window to create "many images"
    for y in range(0, h - TILE_SIZE + 1, TILE_SIZE):
        for x in range(0, w - TILE_SIZE + 1, TILE_SIZE):
            
            # Crop the exact same coordinates from all three images
            crop_a = whole_a.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
            crop_b = whole_b.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
            crop_label = whole_label.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))

            # Apply your enhancement to the RGB images
            crop_a = enhance_crop(crop_a)
            crop_b = enhance_crop(crop_b)

            filename = f"{y // TILE_SIZE}_{x // TILE_SIZE}.tif"

            crop_a.save(os.path.join(out_a, filename))
            crop_b.save(os.path.join(out_b, filename))
            crop_label.save(os.path.join(out_label, filename))
            
            count += 1

    print(f"Successfully processed {split_name}! Created {count} patches using your enhancement process.")

print("Processing train split...")
process_split('train')
print("Processing test split...")
process_split('test')

print("\nDone! The data is now ready for training.")