"""E4c: a baseline with no token order at all, by construction (bag of taxon embeddings).

Each sample becomes ONE 100-d vector: the mean of vocab_embeddings.npy rows over the taxa it contains
(the same table the transformer looks tokens up in). This mirrors Pope's own `use_static_embeddings` mode
(finetune_discriminator/electra_discriminator.py), which averages the same embeddings, optionally weighted by
abundance, and puts a classification head on top.

Two bags x two classifier settings, trained on AGP-IBD, tested on HMP2 and Halfvarson:
  unweighted  plain mean: only WHICH taxa are present          ~ the transformer with shuffled order
  weighted    mean weighted by read count: abundance, no order ~ the transformer with Pope's order
  class_weight balanced / None (the §1.6 rule: every comparison under at least two settings)
LogisticRegression (lbfgs) is deterministic, so there is one fit per setting, not one per seed.
"""
import csv
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pope_model as pm
from _paths import RESULTS
from evaluate import bootstrap_ci

CSV_PATH = RESULTS / "e4c.csv"
COLUMNS = ["bag", "class_weight", "test_set", "auroc", "auroc_ci_lo", "auroc_ci_hi", "auprc", "n_train", "n_test",
           "timestamp"]

def bag_features(arr, emb, weighted):
    """One vector per sample: the mean embedding of the taxa present in it.

    arr      (n, 512, 2) Pope's coordinate format: arr[i, j] = [vocab id, read count]
    emb      (26726, 100) vocab_embeddings.npy; row t is taxon t's embedding
    weighted False: every present taxon counts once (plain mean)
             True:  weight each taxon by its read count / the sample's total reads
    returns  (n, 100) float array
    """
    n = arr.shape[0]
    out = np.zeros((n, emb.shape[1]), dtype=emb.dtype)

    for i in range(n):
        sample = arr[i]

        ids = sample[:, 0].astype(int)
        counts = sample[:, 1].astype(float)

        mask = counts > 0
        if not np.any(mask):
            continue

        valid_ids = ids[mask]
        valid_embs = emb[valid_ids]

        if weighted:
            weights = counts[mask] / counts[mask].sum()
            out[i] = np.average(valid_embs, axis=0, weights=weights)
        else:
            out[i] = np.mean(valid_embs, axis=0)

    return out


def main():
    emb = np.load(pm.path("vocab_embeddings"))
    data = {c: pm.load_cohort(c) for c in ["agp_ibd", *pm.CFG["e4"]["test_cohorts"]]}
    stamp = datetime.now().isoformat(timespec="seconds")
    rows = []
    for weighted in (False, True):
        feats = {c: bag_features(arr, emb, weighted) for c, (arr, _) in data.items()}
        Xtr, ytr = feats["agp_ibd"], data["agp_ibd"][1]
        for cw in ("balanced", None):
            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight=cw, max_iter=5000)).fit(Xtr, ytr)
            for c in pm.CFG["e4"]["test_cohorts"]:
                y = data[c][1]
                p = clf.predict_proba(feats[c])[:, 1]
                (lo, hi), _ = bootstrap_ci(y, p, seed=0)
                rows.append({"bag": "weighted" if weighted else "unweighted", "class_weight": str(cw),
                             "test_set": pm.CFG["cohorts"][c]["name"], "auroc": round(roc_auc_score(y, p), 4),
                             "auroc_ci_lo": round(lo, 4), "auroc_ci_hi": round(hi, 4),
                             "auprc": round(average_precision_score(y, p), 4), "n_train": len(ytr), "n_test": len(y),
                             "timestamp": stamp})
                r = rows[-1]
                print(f"{r['bag']:10s} {r['class_weight']:8s} {r['test_set']:11s} AUROC {r['auroc']:.4f} "
                      f"[{r['auroc_ci_lo']:.3f}, {r['auroc_ci_hi']:.3f}]")
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
