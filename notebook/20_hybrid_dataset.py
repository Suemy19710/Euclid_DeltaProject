import os
import shutil
import random

# Configuration
REAL_DIR = "../einstein_rings/no_lens"
GAN_DIR = "../einstein_rings/generated_no_rings_gan"  
VAE_DIR =  "../einstein_rings/generated_no_rings_vae"
FINAL_DIR =  "../einstein_rings/hybrid_dataset_no_lens"

# Ratios suggested by feedback
NUM_GAN = 1400
NUM_VAE = 600

os.makedirs(FINAL_DIR, exist_ok=True)

def collect_images(source, count, prefix, destination):
    files = [f for f in os.listdir(source) if f.lower().endswith(('.png', '.jpg'))]
    selected = random.sample(files, min(count, len(files)))
    for i, f in enumerate(selected):
        shutil.copy(os.path.join(source, f), os.path.join(destination, f"{prefix}_{i}.png"))
    return len(selected)

# 1. Copy GAN images (70%)
count_gan = collect_images(GAN_DIR, NUM_GAN, "synth_gan", FINAL_DIR)
# 2. Copy VAE images (30%)
count_vae = collect_images(VAE_DIR, NUM_VAE, "synth_vae", FINAL_DIR)
# 3. Copy ALL Real images
count_real = collect_images(REAL_DIR, 195, "real", FINAL_DIR)

print(f"Final Dataset Created: {count_gan} GAN, {count_vae} VAE, {count_real} Real.")
print(f"Total images in {FINAL_DIR}: {count_gan + count_vae + count_real}")