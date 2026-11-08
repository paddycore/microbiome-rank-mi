"""E9 step 1: embed Pope's 26,726 V4 ASVs (140 nt) with Micro16S (Bishop et al. 2026, bioRxiv 10.64898/2026.03.21.713432).

Micro16S maps a 16S fragment of up to 600 bp to a 256-d, L2-normalised vector placed by GTDB phylogeny; the AGP V4
region (136-231 bp with padding) is one of its 29 training regions, so Pope's 140-nt ASVs are in range. The paper's
transformer encodes AGP ASVs with the application model m16s_002, used here too.

The repos (github.com/HaigBishop/micro16s, micro16s-transformer) are cloned outside this repo, at MICRO16S_DIR. They
ship no licence file: use locally for research, do not redistribute the weights. The checkpoint is loaded with
torch.load(weights_only=True) (tensors and plain config only), not with the repo's loader, which unpickles freely.

Output: data/processed/features/m16s_asv_embeddings.npy (26,726 x 256, row = Pope vocab ID), plus a sanity report:
near-identical ASVs must sit closer than random pairs, and embedding distance must track sequence identity.

Usage: python src/e9_micro16s_embed.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

import pope_model as pm

MICRO16S_DIR = Path.home() / "micro16s_work"
CKPT = MICRO16S_DIR / "micro16s/models/m16s_002/ckpts/m16s_002_16000_batches.pth"
OUT = pm.path("features_dir") / "m16s_asv_embeddings.npy"
WINDOW = 200          # the model is padding-invariant (checked below), so a 140-nt ASV needs no 600-wide window


def load_model():
    sys.path.append(str(MICRO16S_DIR / "micro16s-transformer"))      # appended: never shadows this project's modules
    from model import Micro16S
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    return Micro16S.load_model(str(CKPT), checkpoint=ckpt).eval()


def asv_sequences():
    seqs = [line.strip() for line in open(pm.path("vocab_embeddings").with_name("seqs_.07_embed.fasta"))
            if not line.startswith(">")]
    return [s for s in seqs if len(s) > 100]                         # drop the 5-nt <unk> record


def sanity(seqs, emb, n=3000, seed=0):
    """Cosine distance vs Hamming distance on random ASV pairs (all 140 nt, so Hamming = mismatches)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), n, replace=False)
    codes = np.array([[ord(c) for c in seqs[i]] for i in idx], dtype=np.uint8)
    e = emb[idx]
    ham, cos = [], []
    for a in range(0, n, 250):
        h = (codes[a:a + 250, None, :] != codes[None, :, :]).sum(-1)
        c = 1 - e[a:a + 250] @ e.T
        mask = np.arange(n)[None, :] > (np.arange(a, a + len(h))[:, None])
        ham.append(h[mask])
        cos.append(c[mask])
    ham, cos = np.concatenate(ham), np.concatenate(cos)
    print(f"pairs: {len(ham):,}  Spearman(cosine distance, mismatches) = {spearmanr(cos, ham).statistic:.3f}")
    for lo, hi in ((1, 3), (4, 10), (11, 20), (21, 40), (41, 141)):
        k = (ham >= lo) & (ham < hi)
        if k.any():
            print(f"  {lo:3d}-{hi - 1:3d} mismatches: n={k.sum():7,}  cosine distance median {np.median(cos[k]):.3f}")


def main():
    t0 = time.time()
    model = load_model()
    seqs = asv_sequences()
    with torch.no_grad():
        full = model(seqs[:64]).numpy()                              # the trained 600-wide window
        model.max_seq_len = WINDOW
        short = model(seqs[:64]).numpy()
        print(f"padding invariance: max |600-window - {WINDOW}-window| = {np.abs(full - short).max():.1e}", flush=True)
        assert np.abs(full - short).max() < 1e-4
        parts = []
        for i in range(0, len(seqs), 512):
            parts.append(model(seqs[i:i + 512]).numpy())
            print(f"  {min(i + 512, len(seqs)):,}/{len(seqs):,} ({time.time() - t0:.0f}s)", flush=True)
        emb = np.concatenate(parts)
    np.save(OUT, emb.astype(np.float32))
    print(f"{emb.shape} embeddings -> {OUT} ({time.time() - t0:.0f}s); norms {np.linalg.norm(emb, axis=1).min():.3f}-"
          f"{np.linalg.norm(emb, axis=1).max():.3f}")
    sanity(seqs, emb)


if __name__ == "__main__":
    main()
