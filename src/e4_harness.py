"""E4 harness: perturb token order at test time, run Pope's 10-model ensemble, log AUROC per condition.

Every condition is a transform (ids, counts, rng) -> ids applied to the test cohorts (HMP2, Halfvarson) in
Pope's order (src/pope_model.py). The ensemble score is the mean P(IBD) over the 10 models, as in Pope's
begin.py. One CSV row per (condition, test set, repeat) in results/e4.csv; per-model probabilities go to
results/preds/e4_<condition>_r<repeat>_<cohort>.npz. Runs already logged are skipped.

Conditions
  sorted        Pope's order, no perturbation (the baseline every delta is measured against)
  pope_dropout  Pope's test-time input dropout (10% of tokens), for the S4-0 reproduction gate only
  shuffle       random order of all taxa (E4a)                  -> e4_transforms.shuffle_tokens
  ties          random order within tied counts only (E4a)      -> e4_transforms.shuffle_within_ties
  reverse       ascending instead of descending (E4a)           -> e4_transforms.reverse_order
  top<K>        keep the K most abundant taxa (E4d)             -> e4_transforms.top_k

Usage:
  python src/e4_harness.py --gate                      # STOP POINT S4-0
  python src/e4_harness.py --conditions shuffle ties reverse
  python src/e4_harness.py --conditions topk
"""
import argparse
import csv
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import pope_model as pm
from _paths import RESULTS
from evaluate import bootstrap_ci
from tokens import CLS, PAD

E4 = pm.CFG["e4"]
CSV_PATH = RESULTS / "e4.csv"
PRED_DIR = RESULTS / "preds"
COLUMNS = ["condition", "test_set", "repeat", "auroc", "auroc_ci_lo", "auroc_ci_hi", "auprc", "delta_vs_sorted",
           "model_auroc_mean", "model_auroc_sd", "n_test", "n_pos_test", "mean_tokens", "seconds", "timestamp"]


def T():
    """e4_transforms, imported only when a condition needs it (the gate runs before it is written)."""
    import e4_transforms
    return e4_transforms


# name -> (repeats, transform(ids, counts, rng) -> ids)
CONDITIONS = {
    "sorted": (1, lambda ids, c, rng: ids),
    "pope_dropout": (E4["dropout_repeats"], lambda ids, c, rng: pm.pope_test_dropout(ids, rng, E4["pope_test_dropout"])),
    "shuffle": (E4["n_repeats"], lambda ids, c, rng: T().shuffle_tokens(ids, rng)),
    "ties": (E4["n_repeats"], lambda ids, c, rng: T().shuffle_within_ties(ids, c, rng)),
    "reverse": (1, lambda ids, c, rng: T().reverse_order(ids)),
    **{f"top{k}": (1, lambda ids, c, rng, k=k: T().top_k(ids, k)) for k in E4["top_k"]},
}


def check_ids(before, after):
    """Invariants every transform must keep; cheap enough to run on every call."""
    assert after.shape == before.shape and after.dtype == before.dtype, "shape/dtype changed"
    assert (after[:, 0] == CLS).all(), "CLS must stay at position 0"
    n = (after != PAD).sum(1)
    assert all((row[:k] != PAD).all() for row, k in zip(after, n)), "padding must stay at the end"


def completed():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH) as f:
        return {(r["condition"], r["test_set"], r["repeat"]) for r in csv.DictReader(f)}


def append_row(row):
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def sorted_scores(cohort):
    return np.load(PRED_DIR / f"e4_sorted_r0_{cohort}.npz")["probs"].mean(0)


def run(name, repeats, transform, models, table, data):
    done = completed()
    for cohort, (ids, counts, y) in data.items():
        test_set = pm.CFG["cohorts"][cohort]["name"]
        for r in range(repeats):
            if (name, test_set, str(r)) in done:
                continue
            t0 = time.time()
            ids_t = transform(ids, counts, np.random.default_rng(r))
            check_ids(ids, ids_t)
            probs = pm.predict_ensemble(models, ids_t, table)
            PRED_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(PRED_DIR / f"e4_{name}_r{r}_{cohort}.npz", probs=probs, y=y, ids=ids_t)
            p = probs.mean(0)
            (lo, hi), _ = bootstrap_ci(y, p, seed=r)
            per_model = [roc_auc_score(y, q) for q in probs]
            auroc = roc_auc_score(y, p)
            base = auroc if name == "sorted" else roc_auc_score(y, sorted_scores(cohort))
            append_row({"condition": name, "test_set": test_set, "repeat": r, "auroc": round(auroc, 4),
                        "auroc_ci_lo": round(lo, 4), "auroc_ci_hi": round(hi, 4),
                        "auprc": round(average_precision_score(y, p), 4), "delta_vs_sorted": round(auroc - base, 4),
                        "model_auroc_mean": round(np.mean(per_model), 4), "model_auroc_sd": round(np.std(per_model), 4),
                        "n_test": len(y), "n_pos_test": int(y.sum()), "mean_tokens": round(((ids_t != PAD).sum(1) - 1).mean(), 1),
                        "seconds": round(time.time() - t0, 1), "timestamp": datetime.now().isoformat(timespec="seconds")})
            print(f"{name:13s} {test_set:11s} r{r:<2d} AUROC {auroc:.4f} [{lo:.3f}, {hi:.3f}]  delta {auroc - base:+.4f}")


def gate():
    """STOP POINT S4-0: does the rebuilt ensemble reproduce Pope's published cross-study AUROC?"""
    rows = list(csv.DictReader(open(CSV_PATH)))
    print(f"\nS4-0 reproduction gate (tolerance {E4['gate_tolerance']})")
    print(f"{'cohort':11s} {'Pope':>6s} {'sorted':>16s} {'pope_dropout (mean of reps)':>28s} {'per-model mean±sd':>18s}  verdict")
    for cohort in E4["test_cohorts"]:
        name = pm.CFG["cohorts"][cohort]["name"]
        s = next(r for r in rows if r["condition"] == "sorted" and r["test_set"] == name)
        d = [float(r["auroc"]) for r in rows if r["condition"] == "pope_dropout" and r["test_set"] == name]
        pope = E4["pope_auroc"][cohort]
        ok = abs(float(s["auroc"]) - pope) <= E4["gate_tolerance"]
        print(f"{name:11s} {pope:6.3f} {s['auroc']:>7s} [{s['auroc_ci_lo']}, {s['auroc_ci_hi']}] "
              f"{np.mean(d):10.4f} (sd {np.std(d):.4f}, n={len(d)}) {s['model_auroc_mean']:>9s} ± {s['model_auroc_sd']:<6s}  "
              f"{'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=[])
    ap.add_argument("--gate", action="store_true", help="run sorted + pope_dropout and print the S4-0 table")
    args = ap.parse_args()
    names = (["sorted", "pope_dropout"] if args.gate else []) + args.conditions
    if "topk" in names:
        names = [n for n in names if n != "topk"] + [f"top{k}" for k in E4["top_k"]]
    if "sorted" not in names and not (PRED_DIR / "e4_sorted_r0_hmp2.npz").exists():
        names = ["sorted"] + names  # every delta needs the baseline

    data = {}
    for cohort in E4["test_cohorts"]:
        arr, y = pm.load_cohort(cohort)
        ids, counts = pm.pope_order(arr)
        data[cohort] = (ids, counts, y)
    table = pm.input_table()
    models = [pm.load_classifier(d) for d in pm.ensemble_dirs()]
    for n in names:
        repeats, transform = CONDITIONS[n]
        run(n, repeats, transform, models, table, data)
    if args.gate:
        gate()


if __name__ == "__main__":
    main()
