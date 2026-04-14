import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import os
import numpy as np
from PIL import Image

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
DATA_PATH = "../einstein_rings/augmented_dataset"
OUTPUT_DIR = "gan_output"                         
IMAGE_SIZE = 64         
CHANNELS = 1            
Z_DIM = 100             
BATCH_SIZE = 64
LR = 1e-4               
LAMBDA_GP = 10          
CRITIC_ITERATIONS = 5   
EPOCHS = 500           
PATIENCE = 20           # Early stopping patience

os.makedirs(OUTPUT_DIR, exist_ok=True)

# gpu Check
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not detected. Training will be very slow on CPU.")

# ==========================================
# 2. MODEL ARCHITECTURES
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

class Critic(nn.Module):
    def __init__(self, channels, features_d):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Conv2d(channels, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),   
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0), 
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True), 
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.critic(x)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.image_files:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L') 
        if self.transform:
            image = self.transform(image)
        return image, 0  

# ==========================================
# 3. GRADIENT PENALTY FUNCTION
# ==========================================

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = (real * alpha + fake * (1 - alpha)).requires_grad_(True)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

# ==========================================
# 4. INITIALIZATION
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = SimpleDataset(root_dir=DATA_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS, 64).to(DEVICE)
critic = Critic(CHANNELS, 64).to(DEVICE)

opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LR, betas=(0.0, 0.9))

# Early Stopping Variables
best_w_dist = float('inf')
stop_counter = 0
min_delta = 0.001

# ==========================================
# 5. TRAINING LOOP
# ==========================================

gen.train()
critic.train()

print(f"Starting Training: {len(dataset)} images found.")

for epoch in range(EPOCHS):
    epoch_w_distances = []

    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        # --- Train Critic ---
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(DEVICE)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            
            w_dist = torch.mean(critic_real) - torch.mean(critic_fake)
            epoch_w_distances.append(w_dist.item())

            gp = gradient_penalty(critic, real, fake, device=DEVICE)
            loss_critic = (-w_dist + LAMBDA_GP * gp)
            
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # --- Train Generator ---
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    # --- Epoch Summary & Early Stopping ---
    avg_w_dist = abs(np.mean(epoch_w_distances))
    print(f"Epoch [{epoch}/{EPOCHS}] W-Dist: {avg_w_dist:.4f} | Loss G: {loss_gen:.4f}")

    if avg_w_dist < (best_w_dist - min_delta):
        best_w_dist = avg_w_dist
        stop_counter = 0
        torch.save(gen.state_dict(), os.path.join(OUTPUT_DIR, "best_einstein_gen.pth"))
    else:
        stop_counter += 1
        if stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # Save visual samples
    if epoch % 10 == 0:
        with torch.no_grad():
            sample_noise = torch.randn(16, Z_DIM, 1, 1).to(DEVICE)
            fake_samples = gen(sample_noise)
            utils.save_image(fake_samples, os.path.join(OUTPUT_DIR, f"epoch_{epoch}.png"), normalize=True)

torch.save(gen.state_dict(), os.path.join(OUTPUT_DIR, "final_einstein_gen.pth"))
print(f"Training Complete. Files saved to folder: {OUTPUT_DIR}")