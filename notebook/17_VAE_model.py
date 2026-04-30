import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_DIR = "../einstein_rings/no_lens"
OUTPUT_DIR = "vae_output_no_lens"
IMAGE_SIZE = 100
LATENT_DIM = 128      
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 200
PATIENCE = 20          # Early stopping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. DYNAMIC DATA LOADER
# ==========================================
# I use one transform for training (flips/rots) and one for validation
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
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

train_loss_history = []
val_loss_history = []

# ==========================================
# 5. TRAINING LOOP
# ==========================================
best_val_loss = float('inf')
stop_counter = 0

print(f"Starting VAE Training on {DEVICE}...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        recon, mu, logvar = model(batch)
        loss = loss_function(recon, batch, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            recon, mu, logvar = model(batch)
            v_loss = loss_function(recon, batch, mu, logvar)
            val_loss += v_loss.item()

    avg_train = train_loss / len(train_ds)
    avg_val = val_loss / len(val_ds)
    train_loss_history.append(avg_train)
    val_loss_history.append(avg_val)
    print(f"Epoch {epoch} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # # Early Stopping
    # if avg_val < best_val_loss:
    #     best_val_loss = avg_val
    #     stop_counter = 0
    #     torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_vae_model.pth"))
    # else:
    #     stop_counter += 1
    #     if stop_counter >= PATIENCE:
    #         print("Early Stopping Triggered.")
    #         break

    # Save visual samples to see progress
    if epoch % 20 == 0:
        with torch.no_grad():
            sample_img = next(iter(val_loader))[0:8].to(DEVICE)
            recon_img, _, _ = model(sample_img)
            comparison = torch.cat([sample_img, recon_img])
            utils.save_image(comparison, os.path.join(OUTPUT_DIR, f"epoch_{epoch}.png"), nrow=8)
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_einstein_vae.pth"))
print(f"Training finished. Best model saved in {OUTPUT_DIR}")

plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VAE Training and Validation Loss')
plt.legend()
plt.grid(True)

# Save the plot to the output directory
loss_plot_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
plt.savefig(loss_plot_path)
print(f"Training finished. Loss curve saved to {loss_plot_path}")