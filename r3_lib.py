import os
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

class ToMinusOneToOne:
    def __call__(self, x):
        return x * 2.0 - 1.0

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_image_grid(x, path, nrow=8):
    x = (x.clamp(-1, 1) + 1) * 0.5
    utils.save_image(x, path, nrow=nrow)

def sinusoidal_time_embedding(t, dim):
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
    def __init__(self, time_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb):
        return self.net(t_emb)

class QBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gn = nn.GroupNorm(8, ch)
        self.pw1 = nn.Conv2d(ch, ch, 1)
        self.pw2 = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        h = self.pw1(F.silu(self.gn(x)))
        h = self.pw2(torch.tanh(h))
        h = h / (h.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-6)
        return x + h

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_ch, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_ch, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        time_emb_dim=256,
        num_classes=10,
        class_emb_dim=128,
        dropout=0.0,
        use_qblock=False,
    ):
        super().__init__()
        self.time_dim = time_emb_dim
        self.use_qblock = use_qblock

        self.class_emb = nn.Embedding(num_classes, class_emb_dim)
        self.time_mlp = TimeMLP(time_dim=time_emb_dim, out_dim=time_emb_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(time_emb_dim + class_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.down1 = ResBlock(c1, c1, time_emb_dim, dropout)
        self.down2 = ResBlock(c1, c2, time_emb_dim, dropout)
        self.ds1 = Downsample(c2)

        self.down3 = ResBlock(c2, c2, time_emb_dim, dropout)
        self.down4 = ResBlock(c2, c3, time_emb_dim, dropout)
        self.ds2 = Downsample(c3)

        self.mid1 = ResBlock(c3, c3, time_emb_dim, dropout)
        self.mid2 = ResBlock(c3, c3, time_emb_dim, dropout)
        self.qblock = QBlock(c3) if use_qblock else nn.Identity()

        self.us1 = Upsample(c3)
        self.up1 = ResBlock(c3 + c3, c2, time_emb_dim, dropout)
        self.up2 = ResBlock(c2, c2, time_emb_dim, dropout)

        self.us2 = Upsample(c2)
        self.up3 = ResBlock(c2 + c2, c1, time_emb_dim, dropout)
        self.up4 = ResBlock(c1, c1, time_emb_dim, dropout)

        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv2d(c1, in_channels, 3, padding=1)

    def forward(self, x, t, y):
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        y_emb = self.class_emb(y)
        cond = self.cond_proj(torch.cat([t_emb, y_emb], dim=1))

        x = self.in_conv(x)

        s1 = self.down1(x, cond)
        d1 = self.down2(s1, cond)
        x = self.ds1(d1)

        s2 = self.down3(x, cond)
        d2 = self.down4(s2, cond)
        x = self.ds2(d2)

        x = self.mid1(x, cond)
        x = self.mid2(x, cond)
        x = self.qblock(x)

        x = self.us1(x)
        x = torch.cat([x, d2], dim=1)
        x = self.up1(x, cond)
        x = self.up2(x, cond)

        x = self.us2(x)
        x = torch.cat([x, d1], dim=1)
        x = self.up3(x, cond)
        x = self.up4(x, cond)

        return self.out_conv(F.silu(self.out_norm(x)))

class Diffusion:
    def __init__(self, timesteps, beta_start, beta_end, device):
        self.device = device
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0
        )
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    @staticmethod
    def _extract(a, t, x_shape):
        out = a.gather(0, t)
        while len(out.shape) < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_omab * noise, noise

    @torch.no_grad()
    def p_sample_ddpm(self, model, x, t, y):
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_ab_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        eps_pred = model(x, t, y)
        model_mean = sqrt_recip_alpha_t * (x - (betas_t / sqrt_one_minus_ab_t) * eps_pred)

        if (t == 0).all():
            return model_mean

        posterior_var_t = self._extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample_ddpm(self, model, shape, device, y):
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample_ddpm(model, x, t, y)
        return x

    @torch.no_grad()
    def sample_ddim(self, model, shape, device, y, steps=50, eta=0.0):
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        for i in range(steps):
            t = ts[i].expand(shape[0])
            t_prev = ts[i + 1].expand(shape[0]) if i + 1 < steps else torch.zeros_like(t)

            ab_t = self._extract(self.alphas_cumprod, t, x.shape)
            ab_prev = self._extract(self.alphas_cumprod, t_prev, x.shape)

            eps = model(x, t, y)
            x0 = (x - torch.sqrt(1.0 - ab_t) * eps) / torch.sqrt(ab_t)
            x0 = x0.clamp(-1, 1)

            sigma = eta * torch.sqrt(
                (1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev)
            )
            dir_xt = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma ** 2, min=0.0)) * eps
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = torch.sqrt(ab_prev) * x0 + dir_xt + sigma * noise
        return x

def get_dataloader(cfg, train=True):
    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            ToMinusOneToOne(),
        ]
    )
    ds = datasets.FashionMNIST(
        root=cfg.data_root, train=train, download=True, transform=tfm
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=train,
    )
    return dl

def write_loss_csv(path, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss"])
        for r in rows:
            w.writerow(r)

@dataclass
class CFG:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_root: str = "./data"
    image_size: int = 28
    batch_size: int = 128
    num_workers: int = 0
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    base_channels: int = 64
    time_emb_dim: int = 256
    class_emb_dim: int = 128
    dropout: float = 0.0
    lr: float = 2e-4
    epochs: int = 10
    grad_clip: float = 1.0
    log_every: int = 200
    sample_every: int = 1
    out_dir: str = "./runs_r3_q"
    seed: int = 42
    use_qblock: bool = True
