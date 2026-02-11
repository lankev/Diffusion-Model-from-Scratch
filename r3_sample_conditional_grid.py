import os
import torch

from r3_lib import CFG, ensure_dir, Diffusion, ConditionalUNet, save_image_grid

CHECKPOINT_PATH = "./runs_r3_q/ckpt_epoch_010.pt"
OUT_DIR = "./samples_r3"
STEPS = 50
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

    y = torch.arange(0, 10, device=device).repeat_interleave(8)

    with torch.no_grad():
        samples = diffusion.sample_ddim(
            model,
            shape=(80, 1, cfg.image_size, cfg.image_size),
            device=device,
            y=y,
            steps=STEPS,
            eta=ETA
        )

    out_path = os.path.join(OUT_DIR, f"conditional_grid_ddim_{STEPS}steps_eta{ETA}.png")
    save_image_grid(samples, out_path, nrow=10)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
