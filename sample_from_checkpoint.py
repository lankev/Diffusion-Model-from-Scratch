import os
import torch

from diffusion_qblock import (
    CFG,
    UNet,
    Diffusion,
    save_image_grid
)

CHECKPOINT_PATH = "./runs/ckpt_epoch_005.pt"
OUT_DIR = "./samples"
NUM_SAMPLES = 64

def main():
    cfg = CFG()
    device = cfg.device

    print("Device utilisé :", device)

    os.makedirs(OUT_DIR, exist_ok=True)

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

    ckpt = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Checkpoint chargé :", CHECKPOINT_PATH)

    with torch.no_grad():
        samples = diffusion.sample(
            model,
            shape=(NUM_SAMPLES, 1, cfg.image_size, cfg.image_size),
            device=device,
        )

    out_path = os.path.join(OUT_DIR, "generated.png")
    save_image_grid(samples, out_path, nrow=int(NUM_SAMPLES ** 0.5))

    print("Images générées →", out_path)

if __name__ == "__main__":
    main()
