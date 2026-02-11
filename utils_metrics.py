# utils_metrics.py — PSNR / SSIM + sauvegarde grilles

import math
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from skimage.metrics import structural_similarity as ssim

def batch_psnr(x_rec: torch.Tensor, x_ref: torch.Tensor, data_range: float = 1.0):
    """Retourne (PSNR_moyen, liste_psnr) pour un batch [B,1,H,W] en [0,1]."""
    xr = x_rec.detach().cpu().numpy()
    xt = x_ref.detach().cpu().numpy()
    B = xt.shape[0]
    out = []
    for i in range(B):
        mse = np.mean((xr[i] - xt[i]) ** 2)
        if mse <= 1e-12:
            out.append(float("inf"))
        else:
            out.append(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))
    return float(np.mean(out)), out

def batch_ssim(x_rec: torch.Tensor, x_ref: torch.Tensor, data_range: float = 1.0):
    """Retourne (SSIM_moyen, liste_ssim) pour un batch [B,1,H,W] en [0,1]."""
    xr = x_rec.detach().cpu().numpy()
    xt = x_ref.detach().cpu().numpy()
    B = xt.shape[0]
    out = []
    for i in range(B):
        # images 2D pour skimage (canal unique)
        out.append(
            ssim(xt[i, 0], xr[i, 0], data_range=data_range, gaussian_weights=True, win_size=7)
        )
    return float(np.mean(out)), out

def save_comparison_grid(x_ref: torch.Tensor, x_rec: torch.Tensor, base_path: str, nrow: int = 8):
    """Sauvegarde deux grilles : *_inputs.png et *_recons.png"""
    grid_in = make_grid(x_ref.cpu(), nrow=nrow, normalize=True)
    grid_out = make_grid(x_rec.cpu(), nrow=nrow, normalize=True)
    save_image(grid_in,  base_path.replace(".png", "_inputs.png"))
    save_image(grid_out, base_path.replace(".png", "_recons.png"))
