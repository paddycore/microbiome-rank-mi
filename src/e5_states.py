"""E5 step 1: cache every layer's hidden states for the probes, plus the confounder table. STOP POINT S5-0.

Models (one cache file per model x cohort):
  ens1..ens10  the bodies of Pope's 10 fine-tuned classifiers (ensemble.zip), the model that transfers
  pre120       the pretrained checkpoint they were fine-tuned from (E4b: does not transfer with a linear head)
  pre0         the epoch-0 checkpoint, the untrained "random embedding" control (Bendidi et al. 2024)
Inputs: AGP-IBD, HMP2, Halfvarson in Pope's order (src/pope_model.py). Stored: [CLS] and mean-over-taxa vectors of
hidden states 0..5, float16, in data/processed/features/e5_<model>_<cohort>.npz (cls, mean: n x 6 x 200).

S5-0 check (ensemble models, test cohorts): the cached last-layer [CLS], put through the model's own classifier
head, must give the same P(IBD) as E4's Pope-order run (results/preds/e4_sorted_r0_<cohort>.npz). Done in float32
before the float16 cast; prints the largest difference.

Confounders -> results/e5_meta.csv, one row per sample in cache order: cohort, y, read depth (total reads from the
full count matrix, data/processed/<cohort>_X.npy), log10 depth, n_taxa (taxa the model sees, max 512).

Usage: python src/e5_states.py     (about 25 min on CPU; existing files are skipped)
"""
import csv
import time

import numpy as np
import torch

import pope_model as pm
from _paths import PROCESSED, RESULTS

E5 = pm.CFG["e5"]
FEATURES_DIR = pm.path("features_dir")


def state_path(model, cohort):
    return FEATURES_DIR / f"e5_{model}_{cohort}.npz"


def load_states(model, cohort):
    """(cls, mean) float32 arrays of shape (n, 6, 200) for one model and cohort."""
    z = np.load(state_path(model, cohort))
    return z["cls"].astype(np.float32), z["mean"].astype(np.float32)


def models():
    """name -> (loader, classifier or None). Loaders are lazy so skipped models cost nothing."""
    out = {f"ens{i}": (lambda d=d: pm.load_classifier(d)) for i, d in enumerate(pm.ensemble_dirs(), start=1)}
    out["pre120"] = lambda: pm.load_encoder()
    out["pre0"] = lambda: pm.load_encoder(pm.path("pretrained_dir") / E5["untrained_checkpoint"])
    return out


def write_meta(data):
    rows = []
    for cohort, (arr, y, ids) in data.items():
        depth = np.asarray(np.load(PROCESSED / f"{cohort}_X.npy", mmap_mode="r").sum(1), dtype=float)
        n_taxa = (arr[:, :, 1] > 0).sum(1)
        for i in range(len(y)):
            rows.append({"cohort": cohort, "index": i, "y": int(y[i]), "depth": int(depth[i]),
                         "log10_depth": round(float(np.log10(depth[i])), 5), "n_taxa": int(n_taxa[i])})
    with open(RESULTS / "e5_meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for c in E5["cohorts"]:
        arr, y = pm.load_cohort(c)
        data[c] = (arr, y, pm.pope_order(arr)[0])
    write_meta(data)
    table = pm.input_table()
    for name, load in models().items():
        if all(state_path(name, c).exists() for c in data):
            continue
        model = load()
        body = model.electra if hasattr(model, "electra") else model
        for c, (_, y, ids) in data.items():
            t0 = time.time()
            cls, mean = pm.all_layer_features(body, ids, table)
            check = ""
            if name.startswith("ens") and c in pm.CFG["e4"]["test_cohorts"]:
                with torch.no_grad():
                    p = torch.softmax(model.classifier(torch.from_numpy(cls[:, -1:, :])), 1)[:, 1].numpy()
                ref = np.load(RESULTS / "preds" / f"e4_sorted_r0_{c}.npz")["probs"][int(name[3:]) - 1]
                check = f"  S5-0 max |P - E4| = {np.abs(p - ref).max():.2e}"
            np.savez_compressed(state_path(name, c), cls=cls.astype(np.float16), mean=mean.astype(np.float16), y=y)
            print(f"{name:7s} {c:8s} n={len(y):5d} {time.time() - t0:5.0f}s{check}", flush=True)


if __name__ == "__main__":
    main()
