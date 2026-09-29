"""S2 diagnostic: attribute the gap between our `ra` baseline and Pope's published No-DR numbers.

Cross-study protocol only (train all AGP-IBD, test HMP2 and HV), seed 0. Starting from the canonical relative
abundance, each of Pope's preprocessing quirks (DeepMicro/data/convert_data.py) is switched on alone and then all
together, with the spec RF and with DeepMicro's un-weighted RF (class_weight=None, everything else as spec).
Appends every run, including failures, to results/s2_diagnostics.csv.
"""
import csv
import json
import time
import traceback
from datetime import datetime

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from _paths import RESULTS
from data import COHORTS, SRC, VOCAB_SIZE
from evaluate import RF_PARAMS, bootstrap_ci

OUT = RESULTS / "s2_diagnostics.csv"
SEED = 0
PUBLISHED = {"hmp2": 0.657, "hv": 0.530}


def densify_variant(arr, dup="sum", zero_taxon0=False, round4=False):
    """Relative abundance (n, VOCAB_SIZE) float32 with Pope's quirks toggled individually."""
    n, L, _ = arr.shape
    X = np.zeros((n, VOCAB_SIZE), np.float32)
    for s in range(0, n, 1000):
        b = np.asarray(arr[s:s + 1000])
        nb = b.shape[0]
        r, c = np.nonzero(b[:, :, 1] > 0)  # present entries, in position order
        ids = b[r, c, 0].astype(np.int64)
        vals = b[r, c, 1]
        if dup == "sum":
            dense = sp.coo_matrix((vals, (r, ids)), shape=(nb, VOCAB_SIZE)).toarray()
        else:  # last write wins among present entries
            key = r * VOCAB_SIZE + ids
            _, first_rev = np.unique(key[::-1], return_index=True)
            keep = len(key) - 1 - first_rev
            dense = np.zeros((nb, VOCAB_SIZE))
            dense[r[keep], ids[keep]] = vals[keep]
        if zero_taxon0:  # the trailing (0, 0) padding overwrites taxon 0 in every sample that has padding
            has_pad = (b[:, :, 1] > 0).sum(1) < L
            dense[has_pad, 0] = 0
        tot = dense.sum(1, keepdims=True)
        dense /= np.where(tot > 0, tot, 1)
        X[s:s + nb] = np.around(dense, 4) if round4 else dense
    return X


VARIANTS = {
    "canonical": {},
    "dup_overwrite": {"dup": "last"},
    "taxon0_zeroed": {"zero_taxon0": True},
    "round_4dp": {"round4": True},
    "all_pope_quirks": {"dup": "last", "zero_taxon0": True, "round4": True},
}
RF_SETTINGS = {"spec_rf": RF_PARAMS, "unweighted_rf": {**RF_PARAMS, "class_weight": None}}


def in_domain_split_check(w):
    """Patient-leakage probe: Pope's shipped IBD_train/IBD_test split (total_IBD = train ++ test, 6,856 / 1,715)
    vs random stratified splits of the same size. If the shipped split is host-blocked and random splits leak
    same-host samples, the shipped split should score clearly lower."""
    from sklearn.model_selection import train_test_split
    X = densify_variant(np.load(SRC / "total_IBD_otu.npy", mmap_mode="r"))
    y = np.load(SRC / "total_IBD_label.npy").reshape(-1).astype(int)
    idx = np.arange(len(y))
    splits = {"shipped_split": (idx[:6856], idx[6856:])}
    for s in range(3):
        tr, te = train_test_split(idx, test_size=1715, stratify=y, random_state=s)
        splits[f"random_split_{s}"] = (tr, te)
    for name, (tr, te) in splits.items():
        ts = datetime.now().isoformat(timespec="seconds")
        rf = RandomForestClassifier(**RF_PARAMS, random_state=SEED)
        t0 = time.time()
        rf.fit(X[tr], y[tr])
        p = rf.predict_proba(X[te])[:, 1]
        auc, apr = roc_auc_score(y[te], p), average_precision_score(y[te], p)
        (lo, hi), _ = bootstrap_ci(y[te], p, SEED)
        w.writerow([f"in_domain_{name}", "spec_rf", f"AGP-IBD holdout (n_pos={y[te].sum()})", SEED, round(auc, 4),
                    round(lo, 4), round(hi, 4), round(apr, 4), round(auc - 0.717, 4),
                    round(time.time() - t0, 1), "OK", ts, ""])
        print(f"in-domain {name:15s} AUROC {auc:.4f} [{lo:.3f}, {hi:.3f}]  AUPR {apr:.4f}  "
              f"(pos {y[te].sum()}/{len(te)})", flush=True)


def main():
    raw = {k: np.load(SRC / f, mmap_mode="r") for k, (f, _) in COHORTS.items()}
    ys = {k: np.load(SRC / lf).reshape(-1).astype(int) for k, (_, lf) in COHORTS.items()}
    new = not OUT.exists()
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["variant", "rf", "test_set", "seed", "auroc", "auroc_ci_lo", "auroc_ci_hi", "auprc",
                        "delta_vs_published", "fit_seconds", "status", "timestamp", "error"])
        for vname, flags in VARIANTS.items():
            Xs = {k: densify_variant(a, **flags) for k, a in raw.items()}
            for rname, params in RF_SETTINGS.items():
                if vname not in ("canonical", "all_pope_quirks") and rname != "spec_rf":
                    continue  # unweighted RF only for the two end points
                ts = datetime.now().isoformat(timespec="seconds")
                try:
                    rf = RandomForestClassifier(**params, random_state=SEED)
                    t0 = time.time()
                    rf.fit(Xs["agp_ibd"], ys["agp_ibd"])
                    fit_s = round(time.time() - t0, 1)
                    for t in ("hmp2", "hv"):
                        p = rf.predict_proba(Xs[t])[:, 1]
                        auc = roc_auc_score(ys[t], p)
                        (lo, hi), _ = bootstrap_ci(ys[t], p, SEED)
                        w.writerow([vname, rname, t, SEED, round(auc, 4), round(lo, 4), round(hi, 4),
                                    round(average_precision_score(ys[t], p), 4),
                                    round(auc - PUBLISHED[t], 4), fit_s, "OK", ts, ""])
                        print(f"{vname:16s} {rname:13s} {t:4s} AUROC {auc:.4f} [{lo:.3f}, {hi:.3f}]  "
                              f"vs published {PUBLISHED[t]} ({auc - PUBLISHED[t]:+.3f})", flush=True)
                except Exception as e:
                    print(f"{vname} {rname} FAILED {e!r}", flush=True)
                    w.writerow([vname, rname, "", SEED, "", "", "", "", "", "", "FAILED", ts,
                                traceback.format_exc(limit=3).replace("\n", " | ")])
                fh.flush()
        del Xs
        in_domain_split_check(w)


if __name__ == "__main__":
    main()
