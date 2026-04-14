import torch
import torch.nn as nn
from torchvision import utils
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
CHANNELS = 1            
Z_DIM = 100             
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "gan_output/best_einstein_gen.pth" 
OUTPUT_FOLDER = "../einstein_rings/generated_rings_final"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 2. MODEL ARCHITECTURE 
# ==========================================
class Generator(nn.Module):
    def __init__(self, z_dim, channels, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),  
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  
            self._block(features_g * 8, features_g * 4, 4, 2, 1),   
            self._block(features_g * 4, features_g * 2, 4, 2, 1),   
            nn.ConvTranspose2d(features_g * 2, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), 
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)

# ==========================================
# 3. LOADING & GENERATION
# ==========================================

# Initialize the model structure
gen = Generator(Z_DIM, CHANNELS, 64).to(DEVICE)

# Load the saved weights
if os.path.exists(MODEL_PATH):
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    gen.eval() # Set to evaluation mode
    print(f"Model loaded successfully from {MODEL_PATH}")
else:
    print(f"ERROR: Could not find {MODEL_PATH}")
    exit()

print(f"Generating 1000 images in {OUTPUT_FOLDER}...")

with torch.no_grad():
    for i in range(1000):
        # Create random noise
        noise = torch.randn(1, Z_DIM, 1, 1).to(DEVICE)
        
        # Generate image
        fake = gen(noise)
        
        # Save (normalize=True converts -1/1 range back to 0/1 for viewing)
        utils.save_image(fake, f"{OUTPUT_FOLDER}/ring_fake_{i}.png", normalize=True)

print(f"Done! Created 1000 images.")