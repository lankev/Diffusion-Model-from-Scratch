"""
Release 2 — Diffusion Model Training & Denoising Network (PyTorch) — Windows-safe

✅ Fixes included:
- No Lambda in transforms (avoids "Can't pickle local object ... <lambda>" on Windows)
- num_workers defaults to 0 (safe for Windows multiprocessing spawn)
- Full DDPM training + sampling for FashionMNIST (28x28 grayscale)

Run:
  pip install torch torchvision tqdm
  python diffusion_qblock.py

Outputs:
  ./runs/
    - ckpt_epoch_XXX.pt
    - samples_epoch_XXX.png
"""

import os
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm


# ---------------------------
# Windows-safe transform helper (picklable)
# ---------------------------

class ToMinusOneToOne:
    """Map tensor in [0,1] -> [-1,1]. Picklable and Windows-safe (no lambda)."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 - 1.0


# ---------------------------
# Config
# ---------------------------

@dataclass
class CFG:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    data_root: str = "./data"
    image_size: int = 28
    batch_size: int = 128

    # IMPORTANT: Windows-safe default (avoid multiprocessing pickling issues)
    num_workers: int = 0

    # Diffusion
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # Model
    base_channels: int = 64
    time_emb_dim: int = 256
    dropout: float = 0.0

    # Train
    lr: float = 2e-4
    epochs: int = 10
    grad_clip: float = 1.0
    log_every: int = 200
    sample_every: int = 1  # epochs

    # Output
    out_dir: str = "./runs"
    seed: int = 42


# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_image_grid(x: torch.Tensor, path: str, nrow: int = 8):
    """
    x expected in [-1, 1]. Convert to [0, 1] for saving.
    """
    x = (x.clamp(-1, 1) + 1) * 0.5
    utils.save_image(x, path, nrow=nrow)


# ---------------------------
# Diffusion schedule + helpers
# ---------------------------

class Diffusion:
    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device: str):
        self.device = device
        self.timesteps = timesteps

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # For reverse sampling
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Posterior variance
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0
        )
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Sample x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        xt = sqrt_ab * x0 + sqrt_omab * noise
        return xt, noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor):
        """
        One reverse step: x_t -> x_{t-1}
        Model predicts epsilon (noise).
        """
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_ab_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        eps_pred = model(x, t)
        model_mean = sqrt_recip_alpha_t * (x - (betas_t / sqrt_one_minus_ab_t) * eps_pred)

        # If t==0, return mean (no noise)
        if (t == 0).all():
            return model_mean

        posterior_var_t = self._extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape, device: str):
        """
        Start from pure noise and iteratively denoise.
        """
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape):
        """
        Extract a[t] for each batch item and reshape to broadcast to x.
        a: (T,)
        t: (B,)
        returns: (B, 1, 1, 1)
        """
        out = a.gather(0, t)
        while len(out.shape) < len(x_shape):
            out = out.unsqueeze(-1)
        return out


# ---------------------------
# Time Embeddings
# ---------------------------

def sinusoidal_time_embedding(t: torch.Tensor, dim: int):
    """
    Sinusoidal embedding (Transformer-style).
    t: (B,)
    returns: (B, dim)
    """
    device = t.device
    half = dim // 2
    t = t.float()

    freq = math.log(10000) / (half - 1)
    freq = torch.exp(torch.arange(half, device=device) * -freq)
    angles = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeMLP(nn.Module):
    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


# ---------------------------
# U-Net building blocks
# ---------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_ch, out_ch)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    Minimal U-Net for 28x28 grayscale.
    Predicts epsilon (noise) with same shape as input.
    """

    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=256, dropout=0.0):
        super().__init__()
        self.time_dim = time_emb_dim
        self.time_mlp = TimeMLP(time_dim=time_emb_dim, out_dim=time_emb_dim)

        # Input
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # Down
        self.down1 = ResBlock(c1, c1, time_emb_dim, dropout)
        self.down2 = ResBlock(c1, c2, time_emb_dim, dropout)
        self.ds1 = Downsample(c2)

        self.down3 = ResBlock(c2, c2, time_emb_dim, dropout)
        self.down4 = ResBlock(c2, c3, time_emb_dim, dropout)
        self.ds2 = Downsample(c3)

        # Mid
        self.mid1 = ResBlock(c3, c3, time_emb_dim, dropout)
        self.mid2 = ResBlock(c3, c3, time_emb_dim, dropout)

        # Up
        self.us1 = Upsample(c3)
        self.up1 = ResBlock(c3 + c3, c2, time_emb_dim, dropout)
        self.up2 = ResBlock(c2, c2, time_emb_dim, dropout)

        self.us2 = Upsample(c2)
        self.up3 = ResBlock(c2 + c2, c1, time_emb_dim, dropout)
        self.up4 = ResBlock(c1, c1, time_emb_dim, dropout)

        # Output
        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv2d(c1, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x = self.in_conv(x)

        s1 = self.down1(x, t_emb)          # (B, c1, 28, 28)
        d1 = self.down2(s1, t_emb)         # (B, c2, 28, 28)
        x = self.ds1(d1)                   # (B, c2, 14, 14)

        s2 = self.down3(x, t_emb)          # (B, c2, 14, 14)
        d2 = self.down4(s2, t_emb)         # (B, c3, 14, 14)
        x = self.ds2(d2)                   # (B, c3, 7, 7)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.us1(x)                    # (B, c3, 14, 14)
        x = torch.cat([x, d2], dim=1)      # skip
        x = self.up1(x, t_emb)
        x = self.up2(x, t_emb)

        x = self.us2(x)                    # (B, c2, 28, 28)
        x = torch.cat([x, d1], dim=1)      # skip
        x = self.up3(x, t_emb)
        x = self.up4(x, t_emb)

        return self.out_conv(F.silu(self.out_norm(x)))


# ---------------------------
# Data
# ---------------------------

def get_dataloader(cfg: CFG) -> DataLoader:
    tfm = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        ToMinusOneToOne(),  # ✅ no lambda, Windows-safe
    ])

    ds = datasets.FashionMNIST(root=cfg.data_root, train=True, download=True, transform=tfm)

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,          # ✅ Windows-safe default: 0
        pin_memory=(cfg.device == "cuda"),
        drop_last=True,
    )
    return dl


# ---------------------------
# Training
# ---------------------------

def train(cfg: CFG):
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    device = cfg.device
    dl = get_dataloader(cfg)

    diffusion = Diffusion(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        device=device,
    )

    model = UNet(
        in_channels=1,
        base_channels=cfg.base_channels,
        time_emb_dim=cfg.time_emb_dim,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg.epochs}", leave=True)

        running_loss = 0.0

        for i, (x0, _) in enumerate(pbar):
            x0 = x0.to(device)

            # Sample random timesteps t in [0, T)
            t = torch.randint(0, cfg.timesteps, (x0.size(0),), device=device, dtype=torch.long)

            # Forward diffusion
            xt, noise = diffusion.q_sample(x0, t)

            # Predict noise
            noise_pred = model(xt, t)

            # MSE loss
            loss = F.mse_loss(noise_pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % cfg.log_every == 0:
                avg = running_loss / (i + 1)
                pbar.set_postfix(loss=f"{avg:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(cfg.out_dir, f"ckpt_epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "cfg": cfg.__dict__,
            },
            ckpt_path,
        )

        # Sample images
        if epoch % cfg.sample_every == 0:
            model.eval()
            with torch.no_grad():
                samples = diffusion.sample(
                    model,
                    shape=(64, 1, cfg.image_size, cfg.image_size),
                    device=device,
                )
            out_path = os.path.join(cfg.out_dir, f"samples_epoch_{epoch:03d}.png")
            save_image_grid(samples, out_path, nrow=8)

        print(f"Epoch {epoch} done. Saved checkpoint: {ckpt_path}")


# ---------------------------
# Entry
# ---------------------------

if __name__ == "__main__":
    cfg = CFG()
    print("Device:", cfg.device)
    print("Output dir:", cfg.out_dir)
    start = time.time()
    train(cfg)
    print("Finished in", round(time.time() - start, 2), "seconds")
