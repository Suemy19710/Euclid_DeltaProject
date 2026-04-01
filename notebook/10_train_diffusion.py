import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "../einstein_rings/einstein_rings_all"          
OUT_DIR = "../einstein_rings/outputs"        
IMAGE_SIZE = 64             # image size: 64x64
BATCH_SIZE = 4              # smaller batch for CPU / limited GPU
EPOCHS = 50
LEARNING_RATE = 1e-4
TIMESTEPS = 100             # fewer steps for faster training
SAVE_EVERY = 5
NUM_WORKERS = 0             # safer on Windows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# DATASET
# =========================================================
class ImageFolderDataset(Dataset):
    def __init__(self, folder, image_size=64):
        self.paths = []
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

        folder = Path(folder)
        if not folder.exists():
            raise ValueError(f"Dataset folder does not exist: {folder}")

        for p in folder.glob("*"):
            if p.suffix.lower() in valid_exts:
                self.paths.append(str(p))

        self.paths.sort()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img


# =========================================================
# TIME EMBEDDING
# =========================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        if half_dim == 0:
            raise ValueError("Time embedding dimension too small.")

        emb_scale = math.log(10000) / (half_dim - 1) if half_dim > 1 else 1.0
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = time[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# =========================================================
# MODEL BUILDING BLOCKS
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        x = self.conv(x)
        time_emb = self.time_mlp(t)[:, :, None, None]
        x = x + time_emb
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, skip, t):
        x = self.up(x)

        # Safety check for shape mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        time_emb = self.time_mlp(t)[:, :, None, None]
        x = x + time_emb
        return x


# =========================================================
# SIMPLE U-NET FOR DIFFUSION
# =========================================================
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True)
        )

        self.input_conv = DoubleConv(3, 64)

        self.down1 = DownBlock(64, 128, time_emb_dim)    # 64x64 -> 32x32
        self.down2 = DownBlock(128, 256, time_emb_dim)   # 32x32 -> 16x16

        self.bottleneck = DoubleConv(256, 256)

        self.up1 = UpBlock(256, 256, 128, time_emb_dim)  # 16x16 -> 32x32
        self.up2 = UpBlock(128, 128, 64, time_emb_dim)   # 32x32 -> 64x64

        self.output_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)

        x = self.input_conv(x)       # [B, 64, 64, 64]

        x, skip1 = self.down1(x, t)  # x -> [B, 128, 32, 32], skip1 -> [B, 128, 64, 64]
        x, skip2 = self.down2(x, t)  # x -> [B, 256, 16, 16], skip2 -> [B, 256, 32, 32]

        x = self.bottleneck(x)       # [B, 256, 16, 16]

        x = self.up1(x, skip2, t)    # [B, 128, 32, 32]
        x = self.up2(x, skip1, t)    # [B, 64, 64, 64]

        return self.output_conv(x)


# =========================================================
# DIFFUSION SCHEDULE
# =========================================================
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


betas = linear_beta_schedule(TIMESTEPS).to(DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def forward_diffusion_sample(x0, t):
    noise = torch.randn_like(x0)

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x0.shape)

    noisy_image = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_image, noise


# =========================================================
# SAMPLING
# =========================================================
@torch.no_grad()
def sample_timestep(x, t, model):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    predicted_noise = model(x, t)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t[0].item() == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def generate_and_save_samples(model, epoch, n=16):
    model.eval()

    x = torch.randn((n, 3, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)

    for i in reversed(range(TIMESTEPS)):
        t = torch.full((n,), i, device=DEVICE, dtype=torch.long)
        x = sample_timestep(x, t, model)

    x = x.clamp(-1, 1)
    x = (x + 1) / 2  # convert from [-1,1] to [0,1]

    save_path = os.path.join(OUT_DIR, f"sample_epoch_{epoch}.png")
    save_image(x, save_path, nrow=4)
    print(f"Saved generated samples to: {save_path}")


# =========================================================
# TRAINING
# =========================================================
def train():
    dataset = ImageFolderDataset(DATA_DIR, image_size=IMAGE_SIZE)

    if len(dataset) == 0:
        raise ValueError(f"No images found in {DATA_DIR}")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True
    )

    model = SimpleUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {DEVICE}")
    print(f"Number of images: {len(dataset)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")

        for batch in progress_bar:
            batch = batch.to(DEVICE)

            t = torch.randint(0, TIMESTEPS, (batch.shape[0],), device=DEVICE).long()
            x_noisy, noise = forward_diffusion_sample(batch, t)

            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} average loss: {avg_loss:.6f}")

        if epoch % SAVE_EVERY == 0 or epoch == 1:
            model_path = os.path.join(OUT_DIR, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model checkpoint to: {model_path}")

            generate_and_save_samples(model, epoch)

    final_model_path = os.path.join(OUT_DIR, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to: {final_model_path}")
    print("Training finished.")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    train()