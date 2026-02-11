import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from r3_lib import CFG, set_seed, ensure_dir, get_dataloader, Diffusion, ConditionalUNet, save_image_grid, write_loss_csv

def main():
    cfg = CFG()
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    device = cfg.device
    dl = get_dataloader(cfg, train=True)

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

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    global_step = 0
    loss_rows = []
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg.epochs}", leave=True)

        for i, (x0, y) in enumerate(pbar):
            x0 = x0.to(device)
            y = y.to(device)

            t = torch.randint(0, cfg.timesteps, (x0.size(0),), device=device, dtype=torch.long)
            xt, noise = diffusion.q_sample(x0, t)

            noise_pred = model(xt, t, y)
            loss = F.mse_loss(noise_pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()

            global_step += 1
            loss_rows.append((global_step, float(loss.item())))

            if global_step % cfg.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

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

        if epoch % cfg.sample_every == 0:
            model.eval()
            with torch.no_grad():
                y_grid = torch.arange(0, 10, device=device).repeat_interleave(8)
                samples = diffusion.sample_ddim(
                    model,
                    shape=(80, 1, cfg.image_size, cfg.image_size),
                    device=device,
                    y=y_grid,
                    steps=50,
                    eta=0.0
                )
            out_path = os.path.join(cfg.out_dir, f"conditional_grid_epoch_{epoch:03d}.png")
            save_image_grid(samples, out_path, nrow=10)

        print(f"Epoch {epoch} done. Saved: {ckpt_path}")

    csv_path = os.path.join(cfg.out_dir, "loss.csv")
    write_loss_csv(csv_path, loss_rows)

    print("Done in", round(time.time() - start, 2), "seconds")
    print("Loss CSV:", csv_path)

if __name__ == "__main__":
    main()
