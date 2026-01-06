import argparse, yaml, os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import make_loaders
from .utils import set_seed, get_device, ensure_dir, save_checkpoint
from .models.autoencoder import ConvAutoencoder


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(args):
    cfg = load_cfg(args.config)
    set_seed(cfg["seed"])

    device = get_device(cfg.get("device", "cuda_if_available"))
    dl_train, dl_val = make_loaders(cfg)

    ae = ConvAutoencoder(
        in_ch=cfg["data"]["channels"],
        latent_dim=cfg["model"]["seq_hidden"]
    ).to(device)

    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-5)

    out_dir = "results/ae"
    ensure_dir(out_dir)
    epochs = int(args.epochs or cfg.get("ae", {}).get("pretrain_epochs", 10))

    for ep in range(1, epochs + 1):
        # ------------------ TRAIN ------------------
        ae.train()
        pbar = tqdm(dl_train, desc=f"AE Train {ep}")
        tr, n = 0.0, 0

        for imgs, _, _, _ in pbar:
            # imgs: (B, K, C, H, W) -> frames: (B*K, C, H, W)
            B, K = imgs.size(0), imgs.size(1)
            frames = imgs.to(device).reshape(B * K, *imgs.shape[2:])

            _, rec = ae(frames)
            loss = F.l1_loss(rec, frames)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr += loss.item() * frames.size(0)
            n += frames.size(0)
            pbar.set_postfix(l1=f"{tr / max(1, n):.4f}")

        # ------------------ VALID ------------------
        ae.eval()
        vr, vn = 0.0, 0
        with torch.no_grad():
            for imgs, _, _, _ in dl_val:
                B, K = imgs.size(0), imgs.size(1)
                frames = imgs.to(device).reshape(B * K, *imgs.shape[2:])

                _, rec = ae(frames)
                loss = F.l1_loss(rec, frames)

                vr += loss.item() * frames.size(0)
                vn += frames.size(0)

        val_l1 = vr / max(1, vn)
        print(f"Epoch {ep}: val_L1={val_l1:.4f}")

        save_checkpoint({"epoch": ep, "model": ae.state_dict()},
                        os.path.join(out_dir, f"ae_epoch_{ep}.pt"))

    # Save final weights
    save_checkpoint({"epoch": epochs, "model": ae.state_dict()},
                    os.path.join(out_dir, "ae_final.pt"))
    print("Saved final AE to results/ae/ae_final.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    main(args)
