import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import random


# 1. Setup
MODEL_PATH = "vae_output/best_vae_model.pth"
INPUT_DIR = "../einstein_rings/einstein_rings_all"
FINAL_OUTPUT_DIR = "../einstein_rings/generated_rings_vae"
LATENT_DIM = 128       # Size of the "knowledge" bottleneck
IMAGE_SIZE = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-3

TOTAL_TO_GENERATE = 2000
IMAGE_SIZE = 100

os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. DYNAMIC DATA LOADER
# ==========================================
# We use one transform for training (flips/rots) and one for validation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180), # Every angle possible
    transforms.ToTensor(),          # Normalizes to [0, 1]
])

class EinsteinDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        # We tell PyTorch the dataset is "long" so it applies many random augmentations
        return 2000 

    def __getitem__(self, idx):
        real_idx = idx % len(self.image_files)
        img_path = os.path.join(self.root_dir, self.image_files[real_idx])
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img

# ==========================================
# 3. VAE ARCHITECTURE
# ==========================================
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Encoder: 100x100 -> 50x50 -> 25x25
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Bottleneck
        self.fc_mu = nn.Linear(64 * 25 * 25, LATENT_DIM)
        self.fc_logvar = nn.Linear(64 * 25 * 25, LATENT_DIM)
        
        # Decoder
        self.decoder_input = nn.Linear(LATENT_DIM, 64 * 25 * 25)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 50x50
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 100x100
            nn.Sigmoid() # Keeps pixels between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_enc = self.encoder(x)
        mu = self.fc_mu(x_enc)
        logvar = self.fc_logvar(x_enc)
        z = self.reparameterize(mu, logvar)
        z_dec = self.decoder_input(z).view(-1, 64, 25, 25)
        return self.decoder(z_dec), mu, logvar



# ==========================================
# 4. LOSS & TRAINING LOGIC
# ==========================================
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (Accuracy of pixels)
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL Divergence (Regularization of the physics space)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

model = VAE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Data Splitting
full_dataset = EinsteinDataset(INPUT_DIR, transform=train_transform)

# 2. Load Model
model = VAE().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 3. Pre-encode all 195 parents (faster than encoding inside the loop)
print("Encoding original images...")
all_mu = []
with torch.no_grad():
    for i in range(len(full_dataset.image_files)):
        img = full_dataset[i].unsqueeze(0).to(DEVICE)
        _, mu, _ = model(img)
        all_mu.append(mu)

# 4. Generate 2000 Images
print(f"Generating {TOTAL_TO_GENERATE} unique Einstein Rings...")

for i in range(TOTAL_TO_GENERATE):
    with torch.no_grad():
        # Pick two different random parents
        mu1 = random.choice(all_mu)
        mu2 = random.choice(all_mu)
        
        # Random blend factor
        alpha = random.random()
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        
        # Decode
        z_dec = model.decoder_input(z_interp).view(-1, 64, 25, 25)
        generated_img = model.decoder(z_dec)
        
        # Add realistic telescope noise (0.02 is safe, 0.04 is "grainy")
        noise = torch.randn_like(generated_img) * 0.03 
        generated_img = torch.clamp(generated_img + noise, 0, 1)

        utils.save_image(generated_img, f"{FINAL_OUTPUT_DIR}/ring_synth_{i:04d}.png")

    if i % 500 == 0:
        print(f"Progress: {i}/{TOTAL_TO_GENERATE}")

print(f"Success! 2000 images ready in {FINAL_OUTPUT_DIR}")