"""E8 step 2: Pope's model on the T2D metagenomes (notebook/plan_e8.md). Two label-free questions:

  specificity  in T2D patients (none has IBD), does Pope's P(IBD) still track low richness, as it does in the IBD
               test cohorts (E5c: ρ -0.48 / -0.58)? If so, the score reads low diversity, not IBD.
  batch        can a probe on the model's representation tell the 6 delivery batches apart (same study, population,
               disease: a purely technical batch), beyond what read depth and richness alone give?

Profiles from src/e8_metagenome_v4.py, three variants: unique-mapping reads only; EM; EM without ASVs under 2 reads.
Each becomes Pope's input (top 512 by count, Pope's order). Patient-level data stay in ~/diabetes_work; only
aggregates go to results/e8.csv.

Usage: python src/e8_t2d_analysis.py
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pope_model as pm
from _paths import RESULTS
from e5_probe import partial_out
from e5c_signal import spearman_ci
from e8_metagenome_v4 import em

PROFILE_DIR = Path.home() / "diabetes_work/e8"
WORK = Path.home() / "diabetes_work"
VARIANTS = ("unique", "em", "em_min2")
N_BOOT, SEED, N_PERM = 1000, 0, 20


def load_profiles():
    """[(batch, sample, participant, pairs, {variant: {vocab id: count}})]"""
    out = []
    for f in sorted(PROFILE_DIR.glob("*/*.json.gz")):
        r = json.load(gzip.open(f, "rt"))
        unique = {int(a): n for a, n in r["unique"].items()}
        amb = {frozenset(x[1:]): x[0] for x in r["ambiguous"]}
        theta = em(unique, amb)
        out.append((r["batch"], r["sample"], r["participant"], r["pairs"],
                    {"unique": unique, "em": {a: c for a, c in theta.items() if c >= 0.5},
                     "em_min2": {a: c for a, c in theta.items() if c >= 2}}))
    return out


def pope_input(counts):
    """(512, 2) [vocab id, count], the largest 512 counts: the format of Pope's cohort arrays."""
    arr = np.zeros((512, 2))
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:512]
    for i, (a, c) in enumerate(top):
        arr[i] = (a, c)
    return arr


def probe(X, y, C=0.1, n_folds=3):
    """Mean balanced accuracy of a scaled, class-balanced LR probe under stratified CV."""
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in folds.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)).fit(X[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(scores))


def selectivity(X, y):
    rng = np.random.default_rng(SEED)
    score = probe(X, y)
    control = np.mean([probe(X, rng.permutation(y)) for _ in range(N_PERM)])
    return score, control, score - control


def main():
    profiles = load_profiles()
    batches = np.array([p[0] for p in profiles])
    y_batch = pd.factorize(batches, sort=True)[0]
    pairs = np.array([p[3] for p in profiles], float)
    print(f"{len(profiles)} samples, {len(set(p[2] for p in profiles))} participants, batches {dict(zip(*np.unique(batches, return_counts=True)))}")
    table = pm.input_table()
    classifiers = [pm.load_classifier(d) for d in pm.ensemble_dirs()]
    bodies = {f"ens{i}": c.electra for i, c in enumerate(classifiers, 1)}
    bodies["pre120"] = pm.load_encoder()
    bodies["pre0"] = pm.load_encoder(pm.path("pretrained_dir") / pm.CFG["e5"]["untrained_checkpoint"])
    rows, per_sample = [], []
    for v in VARIANTS:
        counts = [p[4][v] for p in profiles]
        richness = np.array([len(c) for c in counts], float)
        v4_depth = np.array([sum(c.values()) for c in counts], float)
        ids, _ = pm.pope_order(np.stack([pope_input(c) for c in counts]))
        p_ibd = pm.predict_ensemble(classifiers, ids, table).mean(0)
        per_sample.append(pd.DataFrame({"variant": v, "batch": batches, "richness": richness, "v4_depth": v4_depth, "p_ibd": p_ibd}))
        for name, x in (("richness", richness), ("log10 V4 reads", np.log10(v4_depth)), ("log10 read pairs", np.log10(pairs))):
            rho, lo, hi = spearman_ci(p_ibd, x, N_BOOT, SEED)
            rows.append(dict(section="specificity", variant=v, what=f"rho(P(IBD), {name})", value=rho, lo=lo, hi=hi))
        rows.append(dict(section="specificity", variant=v, what="median P(IBD)", value=float(np.median(p_ibd)),
                         lo=float(np.percentile(p_ibd, 25)), hi=float(np.percentile(p_ibd, 75))))
        rows.append(dict(section="profile", variant=v, what="median richness", value=float(np.median(richness)),
                         lo=float(richness.min()), hi=float(richness.max())))
        Z = np.c_[np.log10(pairs), np.log10(v4_depth), richness]
        s, c, d = selectivity(Z, y_batch)
        rows.append(dict(section="batch", variant=v, what="probe on depth + richness only", value=d, lo=s, hi=c))
        for model, body in bodies.items():
            cls, mean = pm.all_layer_features(body, ids, table)
            for pooling, X in (("cls", cls[:, -1]), ("mean", mean[:, -1])):
                s, c, d = selectivity(X, y_batch)
                folds = StratifiedKFold(3, shuffle=True, random_state=SEED)
                resid = []
                for tr, te in folds.split(X, y_batch):           # same probe after partialling out depth + richness
                    Xtr, Xte = partial_out(X[tr], X[te], Z[tr], Z[te])
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)).fit(Xtr, y_batch[tr])
                    resid.append(balanced_accuracy_score(y_batch[te], clf.predict(Xte)))
                rows.append(dict(section="batch", variant=v, what=f"probe {model} {pooling} L5", value=d, lo=s, hi=c,
                                 resid_score=float(np.mean(resid))))
        print(f"{v}: done", flush=True)
    d = pd.DataFrame(rows)
    d.round(4).to_csv(RESULTS / "e8.csv", index=False)
    pd.concat(per_sample).to_csv(WORK / "e8_per_sample.csv", index=False)       # patient-level: outside the repo
    print(d.round(3).to_string())


if __name__ == "__main__":
    main()
