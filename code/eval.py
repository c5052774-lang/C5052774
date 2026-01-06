import argparse, yaml, os
import torch
from torchvision.utils import save_image
from .utils import set_seed, get_device, ensure_dir
from .data import make_loaders
from .models.model import MMSequenceModel
import torch.nn.functional as F

def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    cfg = load_cfg(args.config)
    set_seed(cfg['seed'])
    device = get_device(cfg['device'])
    dl_train, dl_val = make_loaders(cfg)
    vocab_size = len(dl_train.dataset.vocab)
    model = MMSequenceModel(cfg, vocab_size=vocab_size).to(device)

    ckpt_dir = cfg['log']['ckpt_dir']
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
    if ckpts:
        state = torch.load(os.path.join(ckpt_dir, ckpts[-1]), map_location=device)
        model.load_state_dict(state['model'])
        print(f"Loaded {ckpts[-1]}")

    model.eval()
    ensure_dir(cfg['log']['sample_dir'])
    with torch.no_grad():
        imgs, caps, img_tgt, cap_tgt = next(iter(dl_val))
        imgs, caps, img_tgt, cap_tgt = imgs.to(device), caps.to(device), img_tgt.to(device), cap_tgt.to(device)
        cap_logits, img_pred, pred_latent = model(imgs, caps, cap_tgt)

        for i in range(min(8, img_pred.size(0))):
            save_image(img_pred[i], os.path.join(cfg['log']['sample_dir'], f'pred_{i}.png'))
            save_image(img_tgt[i], os.path.join(cfg['log']['sample_dir'], f'target_{i}.png'))

        preds = cap_logits.argmax(-1)
        mask = (cap_tgt != 0).float()
        acc = (preds.eq(cap_tgt).float()*mask).sum() / mask.sum().clamp_min(1.0)
        print(f"Caption token accuracy (masked): {acc.item():.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()
    main(args)
