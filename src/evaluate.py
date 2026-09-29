"""Step 6: evaluation harness.

Protocols
  cross    train on all of AGP-IBD (8,571), test on HMP2 and on Halfvarson separately, no fine-tuning
           (Pope et al. §3.2; DeepMicro --use_all_data_for_train). One fit -> one row per test cohort.
  cv       5-fold stratified CV within AGP-IBD (shuffle, random_state=seed). NOT grouped by patient: no subject
           ID ships with the data, and S2 showed same-host leakage, so this is an UPPER BOUND on in-domain
           performance. auroc/auprc and CIs use the pooled out-of-fold predictions; auroc_fold_mean/sd per fold.
  holdout  Pope's shipped split: train on IBD_train rows (total_IBD[:6856]), test on IBD_test rows
           (total_IBD[6856:], 127 IBD / 1,715). No nearest-neighbour label leakage (S2), so this is the
           leakage-free in-domain reference. The split is fixed; seeds vary only the RF.

RF variants (decided at S2): `spec` = the spec's RF (primary); `unweighted` = identical but class_weight=None
(DeepMicro default), a sensitivity analysis. Within a variant the classifier is identical for every encoding.

Every encoding statistic is fitted on the training data of that fit only. Every run is appended to
results/phase0_rf.csv, including failures; runs already logged with status OK are skipped (resumable).

Usage:
  python src/evaluate.py --encodings ra pa rank_within clr rank_raw --seeds 0 1 2 3 4 --protocols cross cv holdout
  python src/evaluate.py --rf-variant unweighted --protocols cross holdout
  python src/evaluate.py --encodings ra_pope --seeds 0 --protocols cross cv   # S2 Pope-faithful diagnostic
"""
import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from _paths import PROCESSED, RESULTS, load_encodings

enc = load_encodings()

RF_PARAMS = dict(n_estimators=500, max_features="sqrt", min_samples_leaf=3,
                 class_weight="balanced", n_jobs=-1)
RF_VARIANTS = {"spec": RF_PARAMS, "unweighted": {**RF_PARAMS, "class_weight": None}}
N_BOOT = 1000
N_FOLDS = 5
N_SHIPPED_TRAIN = 6856  # total_IBD = IBD_train (6,856) ++ IBD_test (1,715), verified at S1
CSV_PATH = RESULTS / "phase0_rf.csv"
PRED_DIR = RESULTS / "preds"
COLUMNS = ["encoding", "train_set", "test_set", "seed", "auroc", "auprc", "auroc_ci_lo", "auroc_ci_hi",
           "n_train", "n_test", "n_pos_test", "n_neg_test", "vocab_size", "mean_ties_per_sample",
           "asv_overlap_frac", "fit_seconds", "status",
           # additions beyond the spec's column list (documented in reports/phase0_log.md)
           "auprc_ci_lo", "auprc_ci_hi", "auroc_fold_mean", "auroc_fold_sd", "asv_overlap_idmatch",
           "data_variant", "rf_params", "timestamp", "error"]
TEST_NAMES = {"hmp2": "HMP2", "hv": "Halfvarson"}
PROTOCOL_TEST_SETS = {"cross": ["HMP2", "Halfvarson"], "cv": ["AGP-IBD-5fold"], "holdout": ["AGP-IBD-holdout"]}


def load(name, variant):
    suffix = "_popefaithful" if variant == "pope_faithful" else ""
    X = np.load(PROCESSED / f"{name}_X{suffix}.npy", mmap_mode="r")
    y = np.load(PROCESSED / f"{name}_y.npy").astype(int)
    return X, y


def encode(name, X_train, X_apply):
    if name == "ra_pope":  # pope_faithful matrices are already relative abundance rounded to 4 dp
        return np.asarray(X_apply, dtype=np.float32), None
    return enc.encode(name, X_train, X_apply)


def bootstrap_ci(y, p, seed):
    rng = np.random.default_rng(seed)
    au, ap = [], []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        au.append(roc_auc_score(yb, p[idx]))
        ap.append(average_precision_score(yb, p[idx]))
    q = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
    return q(au), q(ap)


def append_row(row):
    row.setdefault("rf_params", json.dumps(RF_PARAMS))
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    RESULTS.mkdir(exist_ok=True)
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def completed_runs():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH) as f:
        return {(r["encoding"], r["test_set"], r["seed"], r["data_variant"], r["rf_params"])
                for r in csv.DictReader(f) if r["status"] == "OK"}


def metric_row(base, y, p, seed, ties):
    (a_lo, a_hi), (p_lo, p_hi) = bootstrap_ci(y, p, seed)
    return {**base, "auroc": round(roc_auc_score(y, p), 4), "auprc": round(average_precision_score(y, p), 4),
            "auroc_ci_lo": round(a_lo, 4), "auroc_ci_hi": round(a_hi, 4),
            "auprc_ci_lo": round(p_lo, 4), "auprc_ci_hi": round(p_hi, 4),
            "n_test": len(y), "n_pos_test": int(y.sum()), "n_neg_test": int((y == 0).sum()),
            "mean_ties_per_sample": "" if ties is None else round(float(np.mean(ties)), 3), "status": "OK"}


def fit(X, y, seed, rf_params):
    rf = RandomForestClassifier(**rf_params, random_state=seed)
    t0 = time.time()
    rf.fit(X, y)
    return rf, time.time() - t0


def run_cross(encoding, seed, ctx):
    X, y = load("agp_ibd", ctx["variant"])
    Xtr, _ = encode(encoding, X, X)
    tests = {t: load(t, ctx["variant"]) for t in TEST_NAMES}
    Xte = {t: encode(encoding, X, tests[t][0]) for t in TEST_NAMES}
    rf, fit_s = fit(Xtr, y, seed, ctx["rf_params"])
    ctx["log"](f"  fit {fit_s:.1f}s")
    for t, name in TEST_NAMES.items():
        Xt, ties = Xte[t]
        yt = tests[t][1]
        p = rf.predict_proba(Xt)[:, 1]
        np.savez(PRED_DIR / f"{encoding}_{ctx['tag']}_cross_{t}_seed{seed}.npz", y=yt, p=p)
        c = ctx["manifest"]["cohorts"][t]
        base = dict(encoding=encoding, train_set="AGP-IBD", test_set=name, seed=seed, n_train=len(y),
                    vocab_size=Xtr.shape[1], asv_overlap_frac=c["asv_overlap_frac_pope_blast"],
                    asv_overlap_idmatch=c["asv_overlap_frac_idmatch"], fit_seconds=round(fit_s, 1),
                    data_variant=ctx["variant"], rf_params=json.dumps(ctx["rf_params"]))
        row = metric_row(base, yt, p, seed, ties)
        append_row(row)
        ctx["log"](f"  {name}: AUROC {row['auroc']} [{row['auroc_ci_lo']}, {row['auroc_ci_hi']}]  AUPR {row['auprc']}")


def run_holdout(encoding, seed, ctx):
    X, y = load("agp_ibd", ctx["variant"])
    Xtr_raw = np.asarray(X[:N_SHIPPED_TRAIN])
    Xtr, _ = encode(encoding, Xtr_raw, Xtr_raw)
    Xte, ties = encode(encoding, Xtr_raw, np.asarray(X[N_SHIPPED_TRAIN:]))
    del Xtr_raw
    ytr, yte = y[:N_SHIPPED_TRAIN], y[N_SHIPPED_TRAIN:]
    rf, fit_s = fit(Xtr, ytr, seed, ctx["rf_params"])
    p = rf.predict_proba(Xte)[:, 1]
    np.savez(PRED_DIR / f"{encoding}_{ctx['tag']}_holdout_seed{seed}.npz", y=yte, p=p)
    base = dict(encoding=encoding, train_set="AGP-IBD-shipped-train", test_set="AGP-IBD-holdout", seed=seed,
                n_train=len(ytr), vocab_size=Xtr.shape[1], asv_overlap_frac=1.0, asv_overlap_idmatch=1.0,
                fit_seconds=round(fit_s, 1), data_variant=ctx["variant"], rf_params=json.dumps(ctx["rf_params"]))
    row = metric_row(base, yte, p, seed, ties)
    append_row(row)
    ctx["log"](f"  holdout: AUROC {row['auroc']} [{row['auroc_ci_lo']}, {row['auroc_ci_hi']}]  AUPR {row['auprc']}"
               f"  ({fit_s:.0f}s)")


def run_cv(encoding, seed, ctx):
    X, y = load("agp_ibd", ctx["variant"])
    oof = np.zeros(len(y))
    ties_all = np.zeros(len(y)) if encoding in enc.RANK_ENCODINGS else None
    fold_auc, fit_total, n_train = [], 0.0, []
    for k, (tr, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        Xtr_raw = np.asarray(X[tr])
        Xtr, _ = encode(encoding, Xtr_raw, Xtr_raw)
        Xte, ties = encode(encoding, Xtr_raw, np.asarray(X[te]))
        del Xtr_raw
        rf, fit_s = fit(Xtr, y[tr], seed, ctx["rf_params"])
        fit_total += fit_s
        oof[te] = rf.predict_proba(Xte)[:, 1]
        if ties is not None:
            ties_all[te] = ties
        fold_auc.append(roc_auc_score(y[te], oof[te]))
        n_train.append(len(tr))
        ctx["log"](f"  fold {k}: AUROC {fold_auc[-1]:.4f}  ({fit_s:.0f}s)")
        del Xtr, Xte, rf
    np.savez(PRED_DIR / f"{encoding}_{ctx['tag']}_cv_seed{seed}.npz", y=y, p=oof, fold_auc=fold_auc)
    base = dict(encoding=encoding, train_set="AGP-IBD", test_set="AGP-IBD-5fold", seed=seed,
                n_train=int(np.mean(n_train)), vocab_size=X.shape[1], asv_overlap_frac=1.0, asv_overlap_idmatch=1.0,
                fit_seconds=round(fit_total, 1), data_variant=ctx["variant"], rf_params=json.dumps(ctx["rf_params"]),
                auroc_fold_mean=round(float(np.mean(fold_auc)), 4), auroc_fold_sd=round(float(np.std(fold_auc)), 4))
    row = metric_row(base, y, oof, seed, ties_all)
    append_row(row)
    ctx["log"](f"  AGP 5-fold: pooled AUROC {row['auroc']} [{row['auroc_ci_lo']}, {row['auroc_ci_hi']}]  "
               f"fold mean {row['auroc_fold_mean']} ± {row['auroc_fold_sd']}  AUPR {row['auprc']}")


RUNNERS = {"cross": run_cross, "cv": run_cv, "holdout": run_holdout}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encodings", nargs="+", default=list(enc.ENCODINGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--protocols", nargs="+", default=["cross", "cv", "holdout"], choices=list(RUNNERS))
    ap.add_argument("--rf-variant", default="spec", choices=list(RF_VARIANTS))
    args = ap.parse_args()
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PROCESSED / "manifest.json").read_text())
    rf_params = RF_VARIANTS[args.rf_variant]
    done = completed_runs()

    def log(msg):
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

    for encoding in args.encodings:
        variant = "pope_faithful" if encoding == "ra_pope" else "canonical"
        tag = variant if args.rf_variant == "spec" else f"{variant}_{args.rf_variant}"
        ctx = dict(variant=variant, manifest=manifest, log=log, rf_params=rf_params, tag=tag)
        for seed in args.seeds:
            for proto in args.protocols:
                tests = PROTOCOL_TEST_SETS[proto]
                if all((encoding, t, str(seed), variant, json.dumps(rf_params)) in done for t in tests):
                    log(f"{encoding} seed={seed} {proto} rf={args.rf_variant}: already logged OK, skipping")
                    continue
                log(f"{encoding} seed={seed} {proto} ({variant}, rf={args.rf_variant})")
                try:
                    RUNNERS[proto](encoding, seed, ctx)
                except Exception as e:  # log failures, never drop them
                    log(f"  FAILED: {e!r}")
                    for t in tests:
                        append_row(dict(encoding=encoding, train_set="AGP-IBD", test_set=t, seed=seed,
                                        data_variant=variant, rf_params=json.dumps(rf_params), status="FAILED",
                                        error=traceback.format_exc(limit=3).replace("\n", " | ")))


if __name__ == "__main__":
    sys.exit(main())
