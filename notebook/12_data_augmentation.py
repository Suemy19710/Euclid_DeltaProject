import os
import random
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_FOLDER = "../einstein_rings/einstein_rings_all"
OUTPUT_FOLDER = "../einstein_rings/augmented_dataset"
TOTAL_IMAGES_NEEDED = 1000

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Get list of original images
original_images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.fits'))]

if len(original_images) == 0:
    print(f"Error: No images found in {INPUT_FOLDER}")
    exit()

# Detect size from the first image
with Image.open(os.path.join(INPUT_FOLDER, original_images[0])) as img:
    IMG_WIDTH, IMG_HEIGHT = img.size

print(f"Detected image size: {IMG_WIDTH}x{IMG_HEIGHT}")
print(f"Found {len(original_images)} base images. Generating {TOTAL_IMAGES_NEEDED} augmented versions...")

# ==========================================
# 2. DEFINING THE AUGMENTATION PIPELINE
# ==========================================
# We use RandomAffine because it handles Rotation and Zoom at the same time
# This is much cleaner and prevents the 'diamond' edge look.
augmentation_pipeline = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    
    # 1. Rotate (0-360) and Zoom (1.2x to 1.5x)
    # The zoom 'scale' of 1.2 to 1.5 ensures the black corners are pushed out of frame
    transforms.RandomAffine(
        degrees=360, 
        scale=(1.2, 1.5), 
        fill=0
    ),
    
    # 2. Random Flips (Valid for space data as there is no 'up')
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    
    # 3. Subtle Brightness/Contrast variation (Simulates different telescope exposures)
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    
    # 4. Ensure it stays the original size
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH))
])

# ==========================================
# 3. EXECUTION LOOP
# ==========================================
count = 0
pbar = tqdm(total=TOTAL_IMAGES_NEEDED)

while count < TOTAL_IMAGES_NEEDED:
    # Pick a random image from the original 195
    base_img_name = random.choice(original_images)
    img_path = os.path.join(INPUT_FOLDER, base_img_name)
    
    try:
        # Load image
        with Image.open(img_path).convert('L') as img:
            # Apply transformations
            augmented_img = augmentation_pipeline(img)
            
            # Save the new image
            save_name = f"einstein_ring_aug_{count}.png"
            augmented_img.save(os.path.join(OUTPUT_FOLDER, save_name))
            
            count += 1
            pbar.update(1)
            
    except Exception as e:
        print(f"Skipping image {base_img_name} due to error: {e}")

pbar.close()
print(f"\nSUCCESS: {count} images are now ready in '{OUTPUT_FOLDER}'")
print("You can now use these 200 images to train your GAN or Diffusion Model.")