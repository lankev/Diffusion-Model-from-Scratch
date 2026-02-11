import os
import torch
from r3_lib import CFG, Diffusion, ConditionalUNet, save_image_grid

OUT_DIR = "./release4"
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_NO_QBLOCK = "./runs_r3/ckpt_epoch_010.pt"
CKPT_WITH_QBLOCK = "./runs_r3_q/ckpt_epoch_010.pt"

def load_model(cfg, device, ckpt_path, use_qblock):
    if not os.path.exists(ckpt_path):
        print("Checkpoint introuvable:", ckpt_path)
        return None

    model = ConditionalUNet(
        in_channels=1,
        base_channels=cfg.base_channels,
        time_emb_dim=cfg.time_emb_dim,
        num_classes=10,
        class_emb_dim=cfg.class_emb_dim,
        dropout=cfg.dropout,
        use_qblock=use_qblock,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError:
        print("Incompatibilite checkpoint / architecture, ablation ignoree")
        return None

    model.eval()
    return model

def main():
    cfg = CFG()
    device = cfg.device

    model_no = load_model(cfg, device, CKPT_NO_QBLOCK, use_qblock=False)
    model_q = load_model(cfg, device, CKPT_WITH_QBLOCK, use_qblock=True)

    if model_no is None or model_q is None:
        print("Ablation QBlock non executable")
        return

    diffusion = Diffusion(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        device=device,
    )

    y = torch.zeros(64, device=device, dtype=torch.long)

    x_no = diffusion.sample_ddim(
        model_no, (64, 1, cfg.image_size, cfg.image_size), device, y, steps=50, eta=0.0
    )
    save_image_grid(x_no, f"{OUT_DIR}/ablation_no_qblock.png")

    x_q = diffusion.sample_ddim(
        model_q, (64, 1, cfg.image_size, cfg.image_size), device, y, steps=50, eta=0.0
    )
    save_image_grid(x_q, f"{OUT_DIR}/ablation_with_qblock.png")

if __name__ == "__main__":
    main()
