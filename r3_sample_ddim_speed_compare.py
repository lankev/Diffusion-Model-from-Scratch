import os
import time
import torch

from r3_lib import CFG, ensure_dir, Diffusion, ConditionalUNet, save_image_grid

CHECKPOINT_PATH = "./runs_r3/ckpt_epoch_010.pt"
OUT_DIR = "./samples_r3_speed"
NUM_SAMPLES = 64
LABEL = 0

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

    y = torch.full((NUM_SAMPLES,), LABEL, device=device, dtype=torch.long)

    for steps in [1000, 200, 100, 50, 25, 10]:
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.time()

        with torch.no_grad():
            if steps == 1000:
                samples = diffusion.sample_ddpm(model, (NUM_SAMPLES, 1, cfg.image_size, cfg.image_size), device, y)
            else:
                samples = diffusion.sample_ddim(model, (NUM_SAMPLES, 1, cfg.image_size, cfg.image_size), device, y, steps=steps, eta=0.0)

        torch.cuda.synchronize() if device == "cuda" else None
        dt = time.time() - t0

        out_path = os.path.join(OUT_DIR, f"samples_steps_{steps}.png")
        save_image_grid(samples, out_path, nrow=8)
        print(f"steps={steps} time={dt:.3f}s saved={out_path}")

if __name__ == "__main__":
    main()
