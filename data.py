from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
from PIL import Image
from collections import Counter
import re
from .utils import strip_tags

SPECIAL = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3}

def _tok(s: str):
    return re.findall(r"\w+|\S", s.lower())

def build_vocab(texts, max_vocab=12000):
    cnt = Counter()
    for s in texts:
        cnt.update(_tok(s))
    vocab = dict(SPECIAL)
    for tok, _ in cnt.most_common(max_vocab - len(SPECIAL)):
        if tok not in vocab:
            vocab[tok] = len(vocab)
    inv = {i:t for t,i in vocab.items()}
    return vocab, inv

def encode_text(s: str, vocab, max_len: int):
    ids = [vocab.get(t, SPECIAL["<unk>"]) for t in _tok(s)]
    ids = [SPECIAL["<bos>"]] + ids[:max_len-2] + [SPECIAL["<eos>"]]
    if len(ids) < max_len:
        ids += [SPECIAL["<pad>"]] * (max_len - len(ids))
    return np.int64(ids)

class StoryReasoningNextFrame(Dataset):
    def __init__(self, split, cfg, vocab=None, inv_vocab=None, seed=42):
        self.cfg = cfg
        self.K = cfg['data']['seq_len']
        self.H = cfg['data']['image_size']
        self.max_len = cfg['data']['max_caption_len']
        self.rng = np.random.RandomState(seed)

        repo = cfg['data'].get('hf_repo', 'daniel3303/StoryReasoning')
        self.ds = load_dataset(repo, split=split, streaming=False)

        # Filter usable rows
        self.rows = []
        for i, row in enumerate(self.ds):
            n = int(row.get('frame_count', len(row['images'])))
            if n >= self.K + 1:
                self.rows.append(i)

        # Build vocab on train split if needed
        if vocab is None:
            assert split == "train", "Vocab should be built from the train split."
            texts = []
            for i in self.rows[:2000]:
                s = strip_tags(self.ds[i]['story'])
                texts.append(s)
            self.vocab, self.inv_vocab = build_vocab(texts, max_vocab=cfg['data']['vocab_size'])
        else:
            self.vocab, self.inv_vocab = vocab, inv_vocab

    def __len__(self):
        return len(self.rows)

    def _prep_img(self, pil):
        im = pil.convert("RGB").resize((self.H, self.H))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return np.transpose(arr, (2,0,1))

    def __getitem__(self, idx):
        ridx = self.rows[idx]
        row = self.ds[ridx]
        images = row['images']
        ctx_frames = [self._prep_img(img) for img in images[:self.K]]
        tgt_frame = self._prep_img(images[self.K])

        story_text = strip_tags(row['story'])
        cap_ids = encode_text(story_text, self.vocab, self.max_len)

        ctx_caps = np.full((self.K, self.max_len), SPECIAL["<pad>"], dtype=np.int64)
        return (
            torch.tensor(np.stack(ctx_frames)),   # (K,C,H,H)
            torch.tensor(ctx_caps),               # (K,L)
            torch.tensor(tgt_frame),              # (C,H,H)
            torch.tensor(cap_ids)                 # (L,)
        )

def make_loaders(cfg):
    train = StoryReasoningNextFrame("train", cfg, seed=cfg['seed'])
    val = StoryReasoningNextFrame("test",  cfg, vocab=train.vocab, inv_vocab=train.inv_vocab, seed=cfg['seed']+1)
    dl_train = DataLoader(train, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=0)
    dl_val   = DataLoader(val,   batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=0)
    return dl_train, dl_val
