# StoryReasoning Multimodal Sequence Model — Autoencoder Version

**Dataset:** Oliveira & Matos (2025) StoryReasoning (HF: `daniel3303/StoryReasoning`)  
**Task:** Given first K image frames (and optional text), predict the (K+1) **image** & **caption**.  
**Innovation:** Predict **autoencoder latent** and decode via a frozen AE decoder; add latent MSE loss.

## Quickstart
```bash
# 0) Setup
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 1) (Optional, recommended) Pretrain the AE on frames
python -m src.pretrain_ae --config config.yaml --epochs 10

# 2) Train the sequence model
python -m src.train --config config.yaml

# 3) Evaluate & save samples
python -m src.eval --config config.yaml
```
Outputs appear in `results/checkpoints/`, `results/samples/`, and `results/ae/`.

## Files
- `src/data.py` — loads HF dataset, builds vocab, returns tensors.
- `src/models/*` — encoders, attention, fusion, sequence GRU, caption decoder, AE, etc.
- `src/pretrain_ae.py` — pretrains AE with L1 reconstruction.
- `src/train.py` — trains sequence model with caption CE + image L1 + latent MSE.
- `src/eval.py` — restores latest checkpoint, prints token accuracy, saves images.
- `config.yaml` — hyperparameters (AE enabled).

## Notes
- If HF download is slow, first run `huggingface-cli login` (optional for some mirrors).
- GPU is recommended; CPU will work but be slower.

## Citation
Oliveira, D. A. P., & Matos, D. M. (2025). StoryReasoning Dataset: Using Chain-of-Thought for Scene Understanding and Grounded Story Generation. arXiv:2505.10292.
