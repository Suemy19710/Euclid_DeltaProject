"""
Einstein Ring Synthetic Image Generator — Matches Real Survey Style
=====================================================================
Generates grayscale Einstein ring images with:
  - Speckled salt-and-pepper background noise (like real CCD survey images)
  - Granular Gaussian background texture
  - Faint background stars scattered in the field
  - Realistic PSF blur and detector noise
  - asinh stretch (same as real pipeline)

Requirements:
    pip install numpy Pillow scipy tqdm

Usage:
    python generate_einstein_rings.py --output ./synthetic_rings --count 305
    python generate_einstein_rings.py --output ./synthetic_rings --count 605
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import os
import argparse
from tqdm import tqdm


# ─────────────────────────────────────────────
#  Gravitational lensing physics
# ─────────────────────────────────────────────

def sie_deflection(x, y, b, q, phi):
    q = max(q, 0.05)
    sqrt_q = np.sqrt(1.0 - q ** 2 + 1e-8)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    xr =  cos_p * x + sin_p * y
    yr = -sin_p * x + cos_p * y
    denom = np.sqrt(q ** 2 * xr ** 2 + yr ** 2 + 1e-8)
    ax_r = (b * q / sqrt_q) * np.arctan(sqrt_q * xr / (denom + 1e-8))
    ay_r = (b * q / sqrt_q) * np.arctanh(sqrt_q * yr / (denom + 1e-8))
    ax = cos_p * ax_r - sin_p * ay_r
    ay = sin_p * ax_r + cos_p * ay_r
    return ax, ay


def sersic(x, y, x0, y0, Re, n, q_s, phi_s, flux):
    bn = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n)
    cos_p, sin_p = np.cos(phi_s), np.sin(phi_s)
    dx, dy = x - x0, y - y0
    xr =  cos_p * dx + sin_p * dy
    yr = (-sin_p * dx + cos_p * dy) / (q_s + 1e-6)
    r  = np.sqrt(xr ** 2 + yr ** 2 + 1e-8)
    return np.clip(flux * np.exp(-bn * ((r / Re) ** (1.0 / n) - 1.0)), 0, None)


# ─────────────────────────────────────────────
#  Background stars (faint point sources)
# ─────────────────────────────────────────────

def add_background_stars(image, rng, n_stars_range=(8, 25), psf_sigma=0.8):
    """Scatter faint star-like point sources across the field."""
    size = image.shape[0]
    n_stars = rng.integers(*n_stars_range)
    for _ in range(n_stars):
        sx = rng.integers(0, size)
        sy = rng.integers(0, size)
        # Most stars are faint, a few are brighter
        flux = rng.choice(
            [rng.uniform(0.05, 0.20),   # faint
             rng.uniform(0.20, 0.50),   # medium
             rng.uniform(0.50, 0.90)],  # bright
            p=[0.65, 0.25, 0.10]
        )
        image[sy, sx] += flux

    # Tiny PSF blur to make stars look like point sources (not single pixels)
    image = gaussian_filter(image, sigma=psf_sigma * 0.5)
    return image


# ─────────────────────────────────────────────
#  Realistic CCD-style noise
# ─────────────────────────────────────────────

def add_survey_noise(image, rng):
    """
    Add layered noise matching real Euclid/HST-like survey cutouts:
      1. Granular Gaussian background texture
      2. Poisson shot noise
      3. Salt-and-pepper speckle (bright cosmic-ray-like dots)
    """
    # 1. Gaussian read noise — slightly granular background
    read_sigma = rng.uniform(0.018, 0.035)
    read_noise = rng.normal(0.0, read_sigma, image.shape)

    # 2. Low-frequency background variation (large-scale sky gradient)
    sky_level = rng.uniform(0.02, 0.06)
    sky_variation = gaussian_filter(
        rng.normal(0, sky_level * 0.3, image.shape), sigma=rng.uniform(4, 10)
    )
    image = image + sky_level + sky_variation

    # 3. Poisson shot noise
    image = np.maximum(image, 0)
    shot = rng.poisson(image * 600) / 600.0 - image

    # 4. Salt-and-pepper speckle noise (bright white dots on dark bg)
    speckle = np.zeros_like(image)
    n_speckle = rng.integers(30, 100)
    sx = rng.integers(0, image.shape[1], n_speckle)
    sy = rng.integers(0, image.shape[0], n_speckle)
    speckle_vals = rng.uniform(0.15, 0.80, n_speckle)
    speckle[sy, sx] = speckle_vals
    # Tiny blur so speckles aren't perfectly 1px (looks more natural)
    speckle = gaussian_filter(speckle, sigma=0.4)

    image = image + read_noise + shot + speckle
    return image


# ─────────────────────────────────────────────
#  Full image generation
# ─────────────────────────────────────────────

def generate_ring(size=64, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    pixel_scale = 0.10  # arcsec/pixel

    # Lensing parameters
    b        = rng.uniform(0.5, 1.4)
    lens_q   = rng.uniform(0.45, 1.0)
    lens_phi = rng.uniform(0, np.pi)

    # Source galaxy
    src_x    = rng.uniform(-0.12, 0.12)
    src_y    = rng.uniform(-0.12, 0.12)
    src_Re   = rng.uniform(0.04, 0.20)
    src_n    = rng.uniform(0.5, 3.0)
    src_q    = rng.uniform(0.3, 1.0)
    src_phi  = rng.uniform(0, np.pi)
    src_flux = rng.uniform(0.6, 2.5)

    # Lens galaxy
    lens_Re   = rng.uniform(0.15, 0.50)
    lens_flux = rng.uniform(0.4, 1.8)

    # Coordinate grid
    half = size * pixel_scale / 2.0
    c = np.linspace(-half, half, size)
    X, Y = np.meshgrid(c, c)

    # Lensing ray-trace
    ax, ay = sie_deflection(X, Y, b, lens_q, lens_phi)
    Xs, Ys = X - ax, Y - ay

    # Lensed source arcs
    lensed = sersic(Xs, Ys, src_x, src_y,
                    src_Re, src_n, src_q, src_phi, src_flux)

    # Lens galaxy (bright central ellipse)
    lens_light = sersic(X, Y, 0, 0,
                        lens_Re, 4.0,
                        rng.uniform(0.5, 1.0), lens_phi, lens_flux)

    image = lensed + lens_light

    # PSF convolution
    image = gaussian_filter(image, sigma=rng.uniform(0.7, 1.5))

    # Add background stars scattered in the field
    image = add_background_stars(image, rng,
                                  n_stars_range=(6, 20),
                                  psf_sigma=rng.uniform(0.6, 1.0))

    # Add realistic CCD noise (granular + speckle)
    image = add_survey_noise(image, rng)

    # Clip & normalise
    image = np.clip(image, 0, None)
    vmax = np.percentile(image, 99.8)
    if vmax > 0:
        image /= vmax
    image = np.clip(image, 0, 1)

    return image


def to_grayscale_image(array):
    """Apply asinh stretch and convert to 8-bit grayscale (matches real pipeline)."""
    stretched = np.arcsinh(array * 5.0) / np.arcsinh(5.0)
    stretched = np.clip(stretched, 0, 1)
    return Image.fromarray((stretched * 255).astype(np.uint8), mode="L")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def generate_dataset(output_dir, count, size=64, seed=42, fmt="PNG"):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    ext = "jpg" if fmt.upper() == "JPEG" else "png"

    print(f"\n🔭  Generating {count} survey-style Einstein ring images → {output_dir}\n")

    for i in tqdm(range(count), desc="Rendering"):
        gray = generate_ring(size=size, rng=rng)
        img  = to_grayscale_image(gray)

        fname = f"synthetic_ring_{i+1:05d}.{ext}"
        save_kwargs = {"format": fmt}
        if fmt.upper() == "JPEG":
            save_kwargs["quality"] = int(rng.integers(88, 98))

        img.save(os.path.join(output_dir, fname), **save_kwargs)

    print(f"\n✅  Done! {count} images saved to: {os.path.abspath(output_dir)}")
    print(f"    195 real + {count} synthetic = {195 + count} total images.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate grayscale synthetic Einstein ring images matching real survey data."
    )
    parser.add_argument("--output", type=str, default="./synthetic_einstein_rings",
                        help="Output folder")
    parser.add_argument("--count",  type=int, default=305,
                        help="Number of images (305 → 500 total, 605 → 800 total)")
    parser.add_argument("--size",   type=int, default=64,
                        help="Image size in pixels — match your real data (default: 64)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--format", type=str, default="JPEG",
                        choices=["PNG", "JPEG"],
                        help="Output format (default: JPEG — matches your real data)")
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output,
        count=args.count,
        size=args.size,
        seed=args.seed,
        fmt=args.format,
    )