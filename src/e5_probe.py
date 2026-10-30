"""E5 step 2: linear probes on every layer -- where is cohort identity, and how much is just depth / taxa count?

For each model (ens1..ens10, pre120, pre0; src/e5_states.py), hidden state 0..5, pooling ([CLS] or mean over
taxa) and probe capacity C:

  cohort          which cohort (AGP / HMP2 / Halfvarson) a sample came from; AGP subsampled (config e5.agp_subsample)
  cohort_resid    the same after partialling out log10 read depth and number of taxa from the hidden state
  disease_agp     IBD vs control inside AGP-IBD (random CV: patient leakage, so an upper bound, as in Phase 0)
  transfer        probe trained on all AGP-IBD, AUROC on HMP2 and on Halfvarson (E4b's question, per layer)
  depth_r2, ntaxa_r2   how well a ridge probe reads the two confounders back out of the hidden state (CV R^2)

Layer 0 [CLS] is the same vector for every sample (CLS input is zeros + position 0, before any attention),
so its probes sit at chance by construction; layer 0 is only informative with mean pooling.
Every probe score is balanced accuracy under stratified K-fold CV (same folds for the real task and its control).
control = the same probe on randomly permuted labels (Hewitt & Liang 2019); selectivity = score - control.
Baseline row (model "confounders_only"): the cohort probe on [log10 depth, n_taxa] alone.

"""
import argparse
import csv
import time
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pope_model as pm
from _paths import RESULTS
from e5_states import load_states

E5 = pm.CFG["e5"]
CSV_PATH = RESULTS / "e5.csv"
MODELS = [f"ens{i}" for i in range(1, 11)] + ["pre120", "pre0"]
LAYERS = range(6)
POOLINGS = ("cls", "mean")
COLUMNS = ["model", "layer", "pooling", "C", "task", "score", "control", "selectivity", "n", "seconds", "timestamp"]


def partial_out(X_train, X_test, Z_train, Z_test):
    """Remove what the confounders Z linearly explain from X, with the fit taken from the TRAINING rows only.

    Fit LinearRegression (with intercept) of X_train on Z_train (one regression for all columns of X at once),
    then return (X_train - its prediction, X_test - its prediction), both predictions from that one fit.
    """

    reg = LinearRegression()
    reg.fit(Z_train,X_train)
    X_train_resid = X_train - reg.predict(Z_train)
    X_test_resid = X_test - reg.predict(Z_test)

    return X_train_resid, X_test_resid


def probe_score(X, y, C, seed, Z=None):
    """Mean balanced accuracy of a linear probe under stratified K-fold CV.

    Folds: StratifiedKFold(n_splits=E5["n_folds"], shuffle=True, random_state=seed).
    Probe: make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000)),
           fitted on the training rows of each fold.
    Z:     if given, apply partial_out to X inside each fold (training rows fit it) before the probe.
    Score: balanced_accuracy_score on each held-out fold, averaged over folds.
    """
    skf = StratifiedKFold(n_splits=E5["n_folds"], shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if Z is not None:
          Z_train, Z_test = Z[train_idx], Z[test_idx]
          X_train, X_test = partial_out(X_train, X_test, Z_train, Z_test)

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        scores.append(balanced_accuracy_score(y_test, y_pred))

    return float(np.mean(scores))


def selectivity(X, y, C, seed, Z=None):
    """(score, control, score - control): control is probe_score on labels permuted with
    np.random.default_rng(seed).permutation(y), same folds (same seed), same Z."""

    score = probe_score(X, y, C, seed, Z=Z)
    rng = np.random.default_rng(seed)
    y_control = rng.permutation(y)
    control = probe_score(X, y_control, C, seed, Z=Z)

    return score, control, score - control


def load_meta():
    return pd.read_csv(RESULTS / "e5_meta.csv")


def cohort_rows(meta):
    """Row indices (into the concatenated agp_ibd, hmp2, hv arrays) used for the cohort probes."""
    rng = np.random.default_rng(E5["seed"])
    agp = np.flatnonzero(meta.cohort == "agp_ibd")
    keep = np.sort(rng.choice(agp, size=min(E5["agp_subsample"], len(agp)), replace=False))
    return np.concatenate([keep, np.flatnonzero(meta.cohort != "agp_ibd")])


def layer_matrix(model, layer, pooling):
    """Hidden state `layer` for all samples, rows in e5_meta.csv order."""
    parts = [load_states(model, c)[0 if pooling == "cls" else 1][:, layer] for c in E5["cohorts"]]
    return np.concatenate(parts)


def transfer_auroc(X, meta, C):
    """Probe on all AGP-IBD (balanced), AUROC on each test cohort."""
    tr = (meta.cohort == "agp_ibd").to_numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000))
    clf.fit(X[tr], meta.y[tr])
    return {c: roc_auc_score(meta.y[meta.cohort == c], clf.predict_proba(X[(meta.cohort == c).to_numpy()])[:, 1])
            for c in pm.CFG["e4"]["test_cohorts"]}


def cell(model, layer, pooling, C, meta, rows):
    t0 = time.time()
    X = layer_matrix(model, layer, pooling)
    Z = meta[["log10_depth", "n_taxa"]].to_numpy(float)
    y_cohort = meta.cohort.map({c: i for i, c in enumerate(E5["cohorts"])}).to_numpy()
    agp = (meta.cohort == "agp_ibd").to_numpy()
    seed = E5["seed"]
    out = []
    for task, args in [("cohort", (X[rows], y_cohort[rows], C, seed)),
                       ("cohort_resid", (X[rows], y_cohort[rows], C, seed, Z[rows])),
                       ("disease_agp", (X[agp], meta.y.to_numpy()[agp], C, seed))]:
        s, c, d = selectivity(*args)
        out.append(dict(task=task, score=s, control=c, selectivity=d, n=len(args[1])))
    for c, a in transfer_auroc(X, meta, C).items():
        out.append(dict(task=f"transfer_{c}", score=a, n=int((meta.cohort == c).sum())))
    folds = StratifiedKFold(E5["n_folds"], shuffle=True, random_state=seed)
    for task, target in [("depth_r2", Z[rows, 0]), ("ntaxa_r2", Z[rows, 1])]:
        r2 = cross_val_score(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), X[rows], target,
                             cv=folds.split(X[rows], y_cohort[rows]), scoring="r2").mean()
        out.append(dict(task=task, score=r2, n=len(rows)))
    stamp = datetime.now().isoformat(timespec="seconds")
    return [{**r, "model": model, "layer": layer, "pooling": pooling, "C": C,
             "seconds": round(time.time() - t0, 1), "timestamp": stamp} for r in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    meta = load_meta()
    rows = cohort_rows(meta)
    if args.quick:
        grid = [(m, l, p, E5["C_main"]) for m in ("ens1", "pre0") for l in (0, 5) for p in POOLINGS]
    else:
        grid = [(m, l, p, E5["C_main"]) for m in MODELS for l in LAYERS for p in POOLINGS]
        grid += [(m, l, p, C) for m in ("ens1", "pre120", "pre0") for l in LAYERS for p in POOLINGS
                 for C in E5["C"] if C != E5["C_main"]]
    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(cell)(*g, meta, rows) for g in grid)
    out = [r for res in results for r in res]
    Z = meta[["log10_depth", "n_taxa"]].to_numpy(float)
    y_cohort = meta.cohort.map({c: i for i, c in enumerate(E5["cohorts"])}).to_numpy()
    s, c, d = selectivity(Z[rows], y_cohort[rows], E5["C_main"], E5["seed"])
    out.append(dict(model="confounders_only", layer="", pooling="", C=E5["C_main"], task="cohort", score=s,
                    control=c, selectivity=d, n=len(rows), timestamp=datetime.now().isoformat(timespec="seconds")))
    path = CSV_PATH.with_name("e5_quick.csv") if args.quick else CSV_PATH
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in out)
    print(f"{len(out)} rows -> {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
