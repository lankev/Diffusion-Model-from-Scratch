import os
import torch

from r3_lib import CFG, ensure_dir, Diffusion, ConditionalUNet, save_image_grid

CHECKPOINT_PATH = "./runs_r3/ckpt_epoch_010.pt"
OUT_DIR = "./denoise_steps"
LABEL = 0
FRAMES = 60
ETA = 0.0

def main():
    cfg = CFG()
    device = cfg.device
    ensure_dir(OUT_DIR)

    diffusion = Diffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device)

    model = ConditionalUNet(
        in_channels=1,
        base_channels=cfg.base_channels,
        time_emb_dim=cfg.time_emb_dim,
        num_classes=10,
        class_emb_dim=cfg.class_emb_dim,
        dropout=cfg.dropout,
        use_qblock=cfg.use_qblock,
    ).to(device)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    shape = (1, 1, cfg.image_size, cfg.image_size)
    x = torch.randn(shape, device=device)
    y = torch.tensor([LABEL], device=device, dtype=torch.long)

    ts = torch.linspace(cfg.timesteps - 1, 0, FRAMES, device=device).long()

    for i in range(FRAMES):
        t = ts[i].expand(shape[0])
        t_prev = ts[i + 1].expand(shape[0]) if i + 1 < FRAMES else torch.zeros_like(t)

        ab_t = diffusion._extract(diffusion.alphas_cumprod, t, x.shape)
        ab_prev = diffusion._extract(diffusion.alphas_cumprod, t_prev, x.shape)

        with torch.no_grad():
            eps = model(x, t, y)
            x0 = (x - torch.sqrt(1.0 - ab_t) * eps) / torch.sqrt(ab_t)
            x0 = x0.clamp(-1, 1)

            sigma = ETA * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev))
            dir_xt = torch.sqrt(1.0 - ab_prev - sigma ** 2) * eps
            noise = torch.randn_like(x) if ETA > 0 else torch.zeros_like(x)
            x = torch.sqrt(ab_prev) * x0 + dir_xt + sigma * noise

        out_path = os.path.join(OUT_DIR, f"step_{i:03d}.png")
        save_image_grid(x0, out_path, nrow=1)

    print("Saved frames in:", OUT_DIR)

if __name__ == "__main__":
    main()
