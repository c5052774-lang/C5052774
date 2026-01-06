import argparse, yaml, os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .utils import set_seed, get_device, ensure_dir, AverageMeter, save_checkpoint
from .data import make_loaders
from .models.model import MMSequenceModel

def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    cfg = load_cfg(args.config)
    # robust numeric casting
    cfg['training']['lr'] = float(cfg['training']['lr'])
    cfg['training']['weight_decay'] = float(cfg['training']['weight_decay'])
    cfg['training']['grad_clip'] = float(cfg['training']['grad_clip'])
    set_seed(cfg['seed'])
    device = get_device(cfg['device'])

    dl_train, dl_val = make_loaders(cfg)
    vocab_size = len(dl_train.dataset.vocab)
    model = MMSequenceModel(cfg, vocab_size=vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'],
                            weight_decay=cfg['training']['weight_decay'])

    ensure_dir(cfg['log']['ckpt_dir']); ensure_dir(cfg['log']['sample_dir'])

    for epoch in range(1, int(cfg['training']['epochs'])+1):
        model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(dl_train, desc=f"Train {epoch}")
        for imgs, caps, img_tgt, cap_tgt in pbar:
            imgs, caps, img_tgt, cap_tgt = imgs.to(device), caps.to(device), img_tgt.to(device), cap_tgt.to(device)

            cap_logits, img_pred, pred_latent = model(imgs, caps, cap_tgt)

            cap_loss = F.cross_entropy(cap_logits.view(-1, cap_logits.size(-1)), cap_tgt.view(-1), ignore_index=0)
            img_loss = F.l1_loss(img_pred, img_tgt)
            total = cfg['loss']['caption_ce_weight']*cap_loss + cfg['loss']['image_l1_weight']*img_loss

            if pred_latent is not None:
                with torch.no_grad():
                    true_latent = model.ae.encode(img_tgt)
                latent_loss = F.mse_loss(pred_latent, true_latent)
                total = total + cfg['loss'].get('latent_mse_weight', 1.0)*latent_loss

            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            opt.step()

            loss_meter.update(total.item(), imgs.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.3f}")

        # simple val
        model.eval()
        with torch.no_grad():
            vloss = 0.0; n=0
            for imgs, caps, img_tgt, cap_tgt in dl_val:
                imgs, caps, img_tgt, cap_tgt = imgs.to(device), caps.to(device), img_tgt.to(device), cap_tgt.to(device)
                cap_logits, img_pred, pred_latent = model(imgs, caps, cap_tgt)
                cap_loss = F.cross_entropy(cap_logits.view(-1, cap_logits.size(-1)), cap_tgt.view(-1), ignore_index=0)
                img_loss = F.l1_loss(img_pred, img_tgt)
                loss = cfg['loss']['caption_ce_weight']*cap_loss + cfg['loss']['image_l1_weight']*img_loss
                if pred_latent is not None:
                    true_latent = model.ae.encode(img_tgt)
                    loss = loss + cfg['loss'].get('latent_mse_weight', 1.0)*F.mse_loss(pred_latent, true_latent)
                vloss += loss.item()*imgs.size(0); n+=imgs.size(0)
            vloss /= max(1,n)
        save_checkpoint({'epoch': epoch, 'model': model.state_dict()}, os.path.join(cfg['log']['ckpt_dir'], f"epoch_{epoch}.pt"))
        print(f"Epoch {epoch}: val_loss={vloss:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()
    main(args)
