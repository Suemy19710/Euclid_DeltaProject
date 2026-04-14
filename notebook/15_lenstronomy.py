import numpy as np
import os
import torch
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from PIL import Image

# ==========================================
# 1. SETUP PARAMETERS
# ==========================================
numPix = 64
deltaPix = 0.08  # arcsec per pixel
output_dir = "../einstein_rings/lenstronomy"
os.makedirs(output_dir, exist_ok=True)

# Define the grid (Corrected for latest Lenstronomy)
ra_at_xy_0 = -(numPix - 1) / 2. * deltaPix
dec_at_xy_0 = -(numPix - 1) / 2. * deltaPix
transform_pix2angle = np.array([[1, 0], [0, 1]]) * deltaPix

kwargs_grid = {
    'nx': numPix, 
    'ny': numPix,
    'ra_at_xy_0': ra_at_xy_0, 
    'dec_at_xy_0': dec_at_xy_0,
    'transform_pix2angle': transform_pix2angle
}
pixel_grid = PixelGrid(**kwargs_grid)

# Setup Models
lens_model_list = ['SIE']  # Singular Isothermal Ellipsoid
source_model_list = ['SERSIC_ELLIPSE']
lens_model_class = LensModel(lens_model_list)
source_model_class = LightModel(source_model_list)
imageModel = ImageModel(pixel_grid, PSF(psf_type='NONE'), lens_model_class, source_model_class)

# ==========================================
# 2. NOISE FUNCTION (Step 2)
# ==========================================
def apply_realistic_noise(image):
    """Adds Poisson (shot) noise and Gaussian (background) noise."""
    # Scale image to a realistic photon count (e.g., max intensity 100)
    image = image * (100 / np.max(image))
    
    # Add Poisson Noise
    noisy = np.random.poisson(np.maximum(image, 0)).astype(float)
    
    # Add Gaussian Background Noise (Simulating CCD read noise)
    gauss = np.random.normal(0, 2, image.shape)
    noisy = noisy + gauss
    
    # Normalize back to 0-255 for PNG
    noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min()) * 255
    return noisy.astype(np.uint8)

# ==========================================
# 3. GENERATION LOOP
# ==========================================
print(f"Generating 5000 physical rings in {output_dir}...")

for i in range(5000):
    # Randomize physical parameters so the 5000 images are diverse
    theta_E = np.random.uniform(0.7, 1.6)      # Size of the ring
    e1 = np.random.uniform(-0.2, 0.2)          # Stretch X
    e2 = np.random.uniform(-0.2, 0.2)          # Stretch Y
    source_x = np.random.uniform(-0.2, 0.2)    # Source position
    source_y = np.random.uniform(-0.2, 0.2)
    
    kwargs_lens = [{'theta_E': theta_E, 'e1': e1, 'e2': e2, 'center_x': 0, 'center_y': 0}]
    kwargs_source = [{
        'amp': 1000, 
        'R_sersic': 0.1, 
        'n_sersic': 1, 
        'e1': 0.05, 'e2': 0.05, 
        'center_x': source_x, 'center_y': source_y
    }]
    
    # 1. Create the perfect mathematical ring
    image = imageModel.image(kwargs_lens, kwargs_source)
    
    # 2. Apply the noise to make it look like real telescope data
    final_image = apply_realistic_noise(image)
    
    # 3. Save
    img = Image.fromarray(final_image)
    img.save(os.path.join(output_dir, f"phys_ring_{i}.png"))

    if i % 500 == 0:
        print(f"Progress: {i}/5000 images created.")

print("Done! have a scientifically accurate dataset of 5000 images.")