import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils
from PIL import Image
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_PATH = "../einstein_rings/augmented_dataset" # Folder containing your 1000 images
OUTPUT_DIR = "diffusion_output"                         # Everything will be saved here
IMAGE_SIZE = 100      # Updated to your specific size
CHANNELS = 1
BATCH_SIZE = 16       # Smaller batch for 100x100 to save VRAM
LR = 2e-4
EPOCHS = 500
TIMESTEPS = 300       # Diffusion steps
PATIENCE = 15         # Early stopping patience
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. DIFFUSION SCHEDULER
# ==========================================
def get_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps).to(DEVICE)

betas = get_beta_schedule(TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def forward_diffusion(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alphas_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alphas_t * x_0 + sqrt_one_minus_alphas_t * noise, noise

# ==========================================
# 3. U-NET ARCHITECTURE (Modified for 100x100)
# ==========================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin = nn.Linear(dim, dim)

    def forward(self, t):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float().to(DEVICE) / self.dim))
        pos_enc_a = torch.sin(t.repeat(1, self.dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, self.dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return self.lin(pos_enc)

class UNet100(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = TimeEmbedding(64)
        
        # Down: 100 -> 50 -> 25
        self.down1 = nn.Conv2d(CHANNELS, 64, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # Result: 50x50
        self.down3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # Result: 25x25
        
        # Bridge
        self.bridge = nn.Conv2d(256, 256, 3, padding=1)
        
        # Up: 25 -> 50 -> 100
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.final = nn.Conv2d(64, CHANNELS, 3, padding=1)
        
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t.view(-1, 1)).view(-1, 64, 1, 1)
        
        # Encoder
        x1 = self.relu(self.down1(x) + t_emb) 
        x2 = self.relu(self.down2(x1))
        x3 = self.relu(self.down3(x2))
        
        # Middle
        x_mid = self.relu(self.bridge(x3))
        
        # Decoder with Skip Connections
        x4 = self.relu(self.up1(x_mid) + x2)
        x5 = self.relu(self.up2(x4) + x1)
        return self.final(x5)

# ==========================================
# 4. DATA LOADER
# ==========================================
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.image_files[idx])).convert('L')
        if self.transform: img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = SimpleDataset(DATA_PATH, transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 5. TRAINING & EARLY STOPPING
# ==========================================
model = UNet100().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

best_val_loss = float('inf')
stop_counter = 0

print(f"Starting 100x100 Diffusion Training on {DEVICE}...")

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        t = torch.randint(0, TIMESTEPS, (batch.shape[0],)).to(DEVICE)
        
        x_noisy, noise = forward_diffusion(batch, t)
        predicted_noise = model(x_noisy, t)
        
        loss = criterion(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (batch.shape[0],)).to(DEVICE)
            x_noisy, noise = forward_diffusion(batch, t)
            pred_noise = model(x_noisy, t)
            v_loss = criterion(pred_noise, noise)
            total_val_loss += v_loss.item()

    avg_train = total_train_loss / len(train_loader)
    avg_val = total_val_loss / len(val_loader)
    
    print(f"Epoch {epoch} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

    # # Early Stopping
    # if avg_val < best_val_loss:
    #     best_val_loss = avg_val
    #     stop_counter = 0
    #     torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_diffusion_100px.pth"))
    # else:
    #     stop_counter += 1
    #     if stop_counter >= PATIENCE:
    #         print("Early Stopping Triggered.")
    #         break

    # Save Samples
    if epoch % 20 == 0:
        with torch.no_grad():
            img = torch.randn((1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)
            for i in reversed(range(TIMESTEPS)):
                t_idx = torch.full((1,), i, dtype=torch.long).to(DEVICE)
                p_noise = model(img, t_idx)
                # Denoising math
                img = (img - (betas[i] / sqrt_one_minus_alphas_cumprod[i]) * p_noise) / torch.sqrt(alphas[i])
            
            utils.save_image(img, os.path.join(OUTPUT_DIR, f"epoch_{epoch}.png"), normalize=True)

print("Training finished.")