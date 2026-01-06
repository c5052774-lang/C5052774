import os, random
import torch
import numpy as np
import re

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(pref: str = "cuda_if_available"):
    if pref == "cuda_if_available" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = 0.0; self.count = 0
    def update(self, val, n=1): self.sum += float(val)*n; self.count += n
    @property
    def avg(self): return self.sum / max(1, self.count)

def save_checkpoint(state, path):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)

def strip_tags(text: str):
    t = re.sub(r"(?is)<gdi[^>]*>", " ", text)
    t = re.sub(r"(?is)</gdi>", " ", t)
    t = re.sub(r"(?is)<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
