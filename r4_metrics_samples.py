import os
import csv
import torch
from r3_lib import CFG, Diffusion, ConditionalUNet

OUT_DIR = "./release4"
os.makedirs(OUT_DIR, exist_ok=True)

CKPT = "./runs_r3/ckpt_epoch_010.pt"
DDIM_STEPS = 50
ETA = 0.0
SAMPLES_PER_CLASS = 64


def detect_qblock(state_dict):
    for k in state_dict.keys():
        if k.startswith("qblock."):
            return True
    return False


def stats(x):
    return (
        x.mean().item(),
        x.std().item(),
        x.min().item(),
        x.max().item(),
    )


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

    for cls in range(10):
        y = torch.full((SAMPLES_PER_CLASS,), cls, device=device, dtype=torch.long)
        x = diffusion.sample_ddim(
            model,
            (SAMPLES_PER_CLASS, 1, cfg.image_size, cfg.image_size),
            device,
            y,
            steps=DDIM_STEPS,
            eta=ETA,
        )
        m, s, mi, ma = stats(x)
        rows.append((cls, SAMPLES_PER_CLASS, DDIM_STEPS, ETA, m, s, mi, ma))

    all_x = torch.cat(
        [
            diffusion.sample_ddim(
                model,
                (SAMPLES_PER_CLASS, 1, cfg.image_size, cfg.image_size),
                device,
                torch.full((SAMPLES_PER_CLASS,), i, device=device, dtype=torch.long),
                steps=DDIM_STEPS,
                eta=ETA,
            )
            for i in range(10)
        ],
        dim=0,
    )

    m, s, mi, ma = stats(all_x)
    rows.append(("ALL", all_x.shape[0], DDIM_STEPS, ETA, m, s, mi, ma))

    out_csv = f"{OUT_DIR}/metrics_samples.csv"
    tmp = out_csv + ".tmp"

    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["class", "num_samples", "ddim_steps", "eta", "mean", "std", "min", "max"]
        )
        for r in rows:
            w.writerow(r)

    os.replace(tmp, out_csv)


if __name__ == "__main__":
    main()
