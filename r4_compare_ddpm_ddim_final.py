import os
import csv
import time
import torch
from r3_lib import CFG, Diffusion, ConditionalUNet, save_image_grid

OUT_DIR = "./release4"
os.makedirs(OUT_DIR, exist_ok=True)

CKPT = "./runs_r3/ckpt_epoch_010.pt"
LABEL = 0
DDIM_STEPS = [200, 100, 50, 25, 10]


def detect_qblock(state_dict):
    for k in state_dict.keys():
        if k.startswith("qblock."):
            return True
    return False


def main():
    cfg = CFG()
    device = cfg.device

    diffusion = Diffusion(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        device=device,
    )

    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    use_qblock = detect_qblock(ckpt["model"])

    print("QBlock detecte :", use_qblock)

    model = ConditionalUNet(
        in_channels=1,
        base_channels=cfg.base_channels,
        time_emb_dim=cfg.time_emb_dim,
        num_classes=10,
        class_emb_dim=cfg.class_emb_dim,
        dropout=cfg.dropout,
        use_qblock=use_qblock,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []

    y = torch.full((64,), LABEL, device=device, dtype=torch.long)

    start = time.time()
    x = diffusion.sample_ddpm(
        model, (64, 1, cfg.image_size, cfg.image_size), device, y
    )
    t = time.time() - start
    path = f"{OUT_DIR}/compare_ddpm_steps_1000_label_{LABEL}.png"
    save_image_grid(x, path)
    rows.append(("ddpm", 1000, None, t, path))
    print("Saved:", path, "time=", round(t, 3), "s")

    for steps in DDIM_STEPS:
        start = time.time()
        x = diffusion.sample_ddim(
            model,
            (64, 1, cfg.image_size, cfg.image_size),
            device,
            y,
            steps=steps,
            eta=0.0,
        )
        t = time.time() - start
        path = f"{OUT_DIR}/compare_ddim_steps_{steps}_eta_0.0_label_{LABEL}.png"
        save_image_grid(x, path)
        rows.append(("ddim", steps, 0.0, t, path))
        print("Saved:", path, "time=", round(t, 3), "s")

    csv_path = f"{OUT_DIR}/compare_ddpm_ddim_times.csv"
    tmp = csv_path + ".tmp"

    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "steps", "eta", "time_seconds", "image_path"])
        for r in rows:
            w.writerow(r)

    os.replace(tmp, csv_path)


if __name__ == "__main__":
    main()
