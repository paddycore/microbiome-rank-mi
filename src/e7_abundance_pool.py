"""E7: does abundance-weighted pooling of Pope's token states transfer better? (notebook/plan_e7.md)

E4 showed the model ignores token order, so abundance only enters through which taxa are fed; E4d/SC-2 showed HMP2 does
better with only its most abundant taxa. Here abundance goes in directly, as weights when pooling the last hidden state:

  cls             the [CLS] vector (= E5b)
  mean            mean over the taxon tokens (presence only; [CLS] and padding excluded)
  abund           mean weighted by relative abundance
  abund_shuffled  control: the same weights shuffled among the sample's taxa (same weight shape, wrong taxa)

Heads: Pope's recipe from E5b (train_head, POPE, AGP 80%, 5 seeds, the same checkpoints), on ens1..ens10, pre120, pre0.

Usage: python src/e7_abundance_pool.py --extract     (forward passes, ~25 min on CPU; existing files are skipped)
       python src/e7_abundance_pool.py --heads       (-> results/e7.csv, results/preds/e5b_e7_<pooling>_80.npz)
"""
import argparse
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

import e5b_head as h
import pope_model as pm
from _paths import RESULTS
from e5_states import load_states, models

POOLINGS = ("cls", "mean", "abund", "abund_shuffled")
CSV_PATH = RESULTS / "e7.csv"


def path(model, cohort):
    return pm.path("features_dir") / f"e7_{model}_{cohort}.npz"


def pool(states, counts, rng):
    """{pooling: (b, d)} from last-layer states (b, L, d) and counts (b, L): position 0 is [CLS], count 0 is padding."""
    w = np.asarray(counts, dtype=float).copy()
    w[:, 0] = 0
    present = w > 0
    mean_w = present / present.sum(1, keepdims=True)
    abund_w = w / w.sum(1, keepdims=True)
    shuf_w = np.zeros_like(abund_w)
    for i in range(len(w)):
        idx = np.flatnonzero(present[i])
        shuf_w[i, idx] = rng.permutation(abund_w[i, idx])
    out = {"cls": states[:, 0]}
    for name, weights in (("mean", mean_w), ("abund", abund_w), ("abund_shuffled", shuf_w)):
        out[name] = np.einsum("bl,bld->bd", weights, states)
    return out


@torch.no_grad()
def extract():
    table = pm.input_table()
    data = {c: pm.pope_order(pm.load_cohort(c)[0]) + (pm.load_cohort(c)[1],) for c in pm.CFG["e5"]["cohorts"]}
    bs = pm.CFG["e4"]["batch_size"]
    for name, load in models().items():
        if all(path(name, c).exists() for c in data):
            continue
        model = load()
        body = model.electra if hasattr(model, "electra") else model
        for c, (ids, counts, y) in data.items():
            t0, rng, parts = time.time(), np.random.default_rng(0), []
            for s in range(0, len(ids), bs):
                b = torch.from_numpy(ids[s:s + bs])
                states = body(inputs_embeds=table[b].to(pm.device()), attention_mask=(b != pm.PAD).long().to(pm.device()),
                              output_hidden_states=True).hidden_states[-1].cpu().numpy()
                parts.append(pool(states, counts[s:s + bs], rng))
            out = {k: np.concatenate([p[k] for p in parts]).astype(np.float32) for k in POOLINGS}
            check = np.abs(out["cls"] - load_states(name, c)[0][:, -1]).max()   # must match E5's cache (float16 there)
            np.savez_compressed(path(name, c), y=y, **out)
            print(f"{name:7s} {c:8s} n={len(y):5d} {time.time() - t0:4.0f}s  max |cls - E5 cache| = {check:.1e}", flush=True)


def features(model, pooling):
    """Same shape as e5b_head.body_features: {cohort: (X, y)}."""
    out = {}
    for c in pm.CFG["e5"]["cohorts"]:
        z = np.load(path(model, c))
        out[c] = z[pooling], z["y"].astype(int)
    return out


def cell(model, seed, pooling):
    h.body_features = partial(features, pooling=pooling)    # head_cell looks this up at call time, also in a worker
    return h.head_cell(model, h.POPE, seed, "80")


def heads(jobs):
    rows = []
    for pooling in POOLINGS:
        grid = [(m, s) for m in h.ALL_MODELS for s in h.E5B["seeds"]]
        runs = dict(zip(grid, Parallel(n_jobs=jobs)(delayed(cell)(m, s, pooling) for m, s in grid)))
        ens = h.ensemble_scores(runs)
        h.body_features = partial(features, pooling=pooling)
        n_train, labels = h.labels_of("80")
        rows += h.head_rows(pooling, "80", runs, ens, h.POPE, n_train, labels)
        h.PRED_DIR.mkdir(exist_ok=True)
        h.save_preds(f"e7_{pooling}", "80", ens, labels)          # -> results/preds/e5b_e7_<pooling>_80.npz
        print(f"{pooling}: done", flush=True)
    pd.DataFrame(rows, columns=h.HEAD_COLUMNS).to_csv(CSV_PATH, index=False)
    print(f"{len(rows)} rows -> {CSV_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--heads", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    if args.extract:
        extract()
    if args.heads:
        heads(args.jobs)
