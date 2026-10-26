"""E4b step 1: cache frozen-encoder features of every sample in Pope's order and in shuffled orders.

Encoder: Pope's pretrained discriminator (config e4.encoder_checkpoint, the checkpoint the ensemble was fine-tuned
from), frozen, inputs built exactly as for the ensemble (src/pope_model.py). For each cohort (AGP-IBD, HMP2,
Halfvarson) and each order -- `sorted` (Pope's) and `shuffled` with each seed in e4.e4b_shuffle_seeds, using
e4_transforms.shuffle_tokens -- it stores the last layer's [CLS] vector and the mean over taxa (200-d each).

Output: data/processed/features/e4b_<cohort>_sorted.npz and e4b_<cohort>_shuffled_s<seed>.npz with arrays
cls, mean, y. Existing files are skipped. Read them back with load_features().

Usage: python src/e4b_features.py      (about 15 min on CPU)
"""
import time

import numpy as np

import pope_model as pm
from e4_transforms import shuffle_tokens

E4 = pm.CFG["e4"]
COHORTS = ["agp_ibd", *E4["test_cohorts"]]


def feature_path(cohort, order, seed=None):
    name = f"e4b_{cohort}_sorted.npz" if order == "sorted" else f"e4b_{cohort}_shuffled_s{seed}.npz"
    return pm.path("features_dir") / name


def load_features(cohort, order, seed=None):
    """(cls, mean, y) for one cohort in one order. `seed` picks the permutation and is ignored for `sorted`."""
    z = np.load(feature_path(cohort, order, seed))
    return z["cls"], z["mean"], z["y"]


def main():
    pm.path("features_dir").mkdir(parents=True, exist_ok=True)
    encoder, table = pm.load_encoder(), pm.input_table()
    for cohort in COHORTS:
        arr, y = pm.load_cohort(cohort)
        ids, _ = pm.pope_order(arr)
        jobs = [("sorted", None, ids)] + [("shuffled", s, None) for s in E4["e4b_shuffle_seeds"]]
        for order, seed, ids_o in jobs:
            out = feature_path(cohort, order, seed)
            if out.exists():
                continue
            if ids_o is None:
                ids_o = shuffle_tokens(ids, np.random.default_rng(seed))
            t0 = time.time()
            cls, mean = pm.encoder_features(encoder, ids_o, table)
            np.savez_compressed(out, cls=cls.astype(np.float32), mean=mean.astype(np.float32), y=y)
            print(f"{cohort:8s} {order:8s} seed={seed}  n={len(y)}  {time.time() - t0:.0f}s -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
