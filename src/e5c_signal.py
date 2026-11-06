"""E5c: what does the model's score track? Community-level signals (richness, read depth) against what is left over.

Inputs, all row-aligned with results/e5_meta.csv (the loader checks every label vector against it):
  results/e5_meta.csv                       cohort, index, y, depth, log10_depth, n_taxa (taxa the model sees, max 512)
  data/raw/.../<cohort>_otu.npy             Pope's full coordinate lists (up to 1,897 entries per sample):
                                              richness     = entries with count > 0, i.e. n_taxa without the 512 cap
                                              richness_ids = distinct vocab IDs (Halfvarson maps several ASVs onto one ID:
                                                             median 361 entries but 281 IDs)
  results/preds/e4_sorted_r0_<cohort>.npz   Pope's 10 heads (E4), source "ensemble" = their mean P(IBD)
  results/preds/{ra,pa}_canonical_cross_<cohort>_seed*.npz   Phase 0 random forests, sources "rf_ra", "rf_pa" (the control:
                                              if an RF looks the same, it is a property of the task, not of the transformer)
  results/preds/e5b_pope_80.npz             extension, sources "e5b_epoch1" / "e5b_epoch50": E5b heads stopped early vs
                                              trained long. Does the score's link to richness change with training length?
  results/e4.csv                            top-K and random-K AUROC with mean_tokens (SC-2)

SC-0  AUROC of richness, n_taxa, richness_ids and log10 depth for IBD, per cohort, with direction.
SC-1  per test cohort and source: nested logistic models M0 (richness + depth), M1 (M0 + score), M2 (score alone), all
      cross-fitted; ΔAUROC(M1 - M0) = what the score adds over the simple signals, ΔAUROC(M1 - M2) = what they add over
      the score (plan_e5c.md's M3 is the same model as M1); AUROC of the cross-fitted residual score; AUROC inside
      richness tertiles; the spread over the source's members. Repeated with n_taxa in place of richness.
SC-2  top-K vs random-K at equal token counts (E4d): the gap is the effect of which taxa, the slope that of how many.
SC-3  Spearman ρ between the ensemble and the RFs against ρ between two ensemble members; ρ of every score with
      richness and depth.

Every AUROC carries a bootstrap CI and cohorts are never pooled (42 and 54 controls). Wording: everything but SC-2 is
correlational, so "the score is explained by X", never "the model uses X".

Usage: python src/e5c_signal.py --sc 0      (SC-0 first: if the direction differs between cohorts, stop and rethink)
       python src/e5c_signal.py --sc 1 2 3
"""
import argparse
import time
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pope_model as pm
from _paths import PROCESSED, RESULTS
from data import COHORTS as RAW_FILES

E4, E5C = pm.CFG["e4"], pm.CFG["e5c"]
COHORTS = pm.CFG["e5"]["cohorts"]
TEST_COHORTS = E4["test_cohorts"]
NAMES = {c: pm.CFG["cohorts"][c]["name"] for c in COHORTS}
PRED_DIR = RESULTS / "preds"
META_CSV, CSV_PATH = RESULTS / "e5c_meta.csv", RESULTS / "e5c.csv"
SC0_VARIABLES = ["richness", "n_taxa", "richness_ids", "log10_depth"]
COVARIATES = {"richness": ["richness", "log10_depth"],        # main: true richness
              "n_taxa": ["n_taxa", "log10_depth"]}            # sensitivity: richness as the model sees it (capped at 512)
SC1_SOURCES = ["ensemble", "rf_ra", "e5b_epoch1", "e5b_epoch50"]
COLUMNS = ["section", "cohort", "source", "covariates", "metric", "K", "tokens", "bin", "strata", "value", "lo", "hi",
           "interval", "n", "n_pos", "timestamp"]


def auroc_ci(y, s, n_boot, seed):
    """(AUROC, lo, hi) as floats: roc_auc_score(y, s) and the 2.5 / 97.5 percentiles of its bootstrap distribution.

    Resampling: rng = np.random.default_rng(seed); each of the n_boot draws is rng.integers(0, n, n) (n row indices
    with replacement, drawn in that order); a draw that holds only one class is skipped, not redrawn.
    """
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    n, boots = len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(roc_auc_score(y, s)), float(lo), float(hi)


def delta_auroc_ci(y, a, b, n_boot, seed):
    """(AUROC(a) - AUROC(b), lo, hi), paired: every draw scores a and b on the SAME rows. Same resampling as auroc_ci."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    n, boots = len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        boots.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(roc_auc_score(y, a) - roc_auc_score(y, b)), float(lo), float(hi)


def crossfit_proba(y, Z, n_folds, seed):
    """Out-of-fold P(y = 1) for every row from make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)) on
    the columns of Z (n, k). Folds: StratifiedKFold(n_folds, shuffle=True, random_state=seed) on y; each row's value
    comes from the model fitted on the other folds."""
    y, Z = np.asarray(y), np.asarray(Z, dtype=float)
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(Z, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Z[tr], y[tr])
        p[te] = model.predict_proba(Z[te])[:, 1]
    return p


def crossfit_residual(y, s, Z, n_folds, seed):
    """Out-of-fold residual s - ŝ for every row, where ŝ = LinearRegression of s on Z fitted on the other folds.
    Same folds as crossfit_proba (stratified on y)."""
    y, s, Z = np.asarray(y), np.asarray(s, dtype=float), np.asarray(Z, dtype=float)
    r = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(Z, y):
        r[te] = s[te] - LinearRegression().fit(Z[tr], s[tr]).predict(Z[te])
    return r


def stratified_auroc(y, s, strata, n_bins, n_boot, seed):
    """AUROC of s inside each of n_bins equal-size bins of `strata`, lowest bin first.

    Bins: np.array_split(np.argsort(strata, kind="stable"), n_bins). Returns one dict per bin:
      {"bin": 0.., "lo_strata": min of strata in the bin, "hi_strata": max, "n": rows, "n_pos": positives,
       "auroc", "lo", "hi": auroc_ci(y_bin, s_bin, n_boot, seed), or all three NaN if the bin holds one class}
    """
    y, s, strata = np.asarray(y), np.asarray(s), np.asarray(strata)
    out = []
    for b, idx in enumerate(np.array_split(np.argsort(strata, kind="stable"), n_bins)):
        yb, sb = y[idx], s[idx]
        if yb.min() == yb.max():
            auc = lo = hi = np.nan
        else:
            auc, lo, hi = auroc_ci(yb, sb, n_boot, seed)
        out.append({"bin": b, "lo_strata": strata[idx].min(), "hi_strata": strata[idx].max(), "n": len(idx),
                    "n_pos": int(yb.sum()), "auroc": auc, "lo": lo, "hi": hi})
    return out


def spearman_ci(a, b, n_boot, seed):
    """(rho, lo, hi): scipy's spearmanr(a, b) statistic and the 2.5 / 97.5 percentiles over n_boot resampled pairs
    (same draws as auroc_ci, with no class check)."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    boots = [spearmanr(a[i], b[i]).statistic for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(spearmanr(a, b).statistic), float(lo), float(hi)


def _per_row(f, n, chunk=1000):
    return np.concatenate([f(slice(s, min(s + chunk, n))) for s in range(0, n, chunk)])


def load_meta():
    """results/e5_meta.csv plus richness and richness_ids, cached in results/e5c_meta.csv."""
    if META_CSV.exists():
        return pd.read_csv(META_CSV)
    meta = pd.read_csv(RESULTS / "e5_meta.csv")
    entries, ids = [], []
    for c in COHORTS:
        arr = np.load(pm.path("cohorts_dir") / RAW_FILES[c][0], mmap_mode="r")
        X = np.load(PROCESSED / f"{c}_X.npy", mmap_mode="r")
        entries.append(_per_row(lambda s: (np.asarray(arr[s, :, 1]) > 0).sum(1), len(arr)))
        ids.append(_per_row(lambda s: (np.asarray(X[s]) > 0).sum(1), len(X)))
    meta["richness"], meta["richness_ids"] = np.concatenate(entries), np.concatenate(ids)
    # the 512-entry input is the top of the same list, so capping richness must give back n_taxa row for row
    assert np.array_equal(np.minimum(meta.richness, 512), meta.n_taxa), "richness is not row-aligned with e5_meta"
    meta.to_csv(META_CSV, index=False)
    return meta


def scores(cohort, y):
    """{source: (score, members)}: P(IBD) per row of the cohort in e5_meta order, and the models averaged into it."""
    def aligned(labels, what):
        assert np.array_equal(np.asarray(labels).astype(int), y), f"{what} is not row-aligned with e5_meta"

    out = {}
    z = np.load(PRED_DIR / f"e4_sorted_r0_{cohort}.npz")
    aligned(z["y"], "E4 ensemble")
    out["ensemble"] = z["probs"].mean(0), list(z["probs"])
    for enc in ("ra", "pa"):
        members = []
        for f in sorted(PRED_DIR.glob(f"{enc}_canonical_cross_{cohort}_seed*.npz")):
            z = np.load(f)
            aligned(z["y"], f.name)
            members.append(z["p"])
        out[f"rf_{enc}"] = np.mean(members, axis=0), members
    path = PRED_DIR / "e5b_pope_80.npz"
    if path.exists():
        z = np.load(path)
        aligned(z[f"{cohort}_y"], path.name)
        for e in E5C["e5b_epochs"]:
            p = z[f"{cohort}_p"][:, z["epochs"].tolist().index(e)]        # (seeds, n): one ensemble of 10 heads per seed
            out[f"e5b_epoch{e}"] = p.mean(0), list(p)
    return out


def row(section, cohort, metric, value, lo="", hi="", interval="", **extra):
    r = {"section": section, "cohort": NAMES.get(cohort, cohort), "metric": metric, "value": value, "lo": lo, "hi": hi,
         "interval": interval, "timestamp": datetime.now().isoformat(timespec="seconds"), **extra}
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}


CI = "bootstrap 95%"


def sc0(meta):
    rows = []
    for c in COHORTS:
        m = meta[meta.cohort == c]
        y = m.y.to_numpy()
        for v in SC0_VARIABLES:
            auc, lo, hi = auroc_ci(y, m[v].to_numpy(float), E4["n_boot"], E5C["seed"])
            rows.append(row("SC-0", c, "auroc", auc, lo, hi, CI, source=v, n=len(y), n_pos=int(y.sum())))
    return rows


def sc1_one(y, Z, score, members):
    """Rows for one cohort x source x covariate set (cohort and labels are filled in by the caller)."""
    f, s, b = E5C["n_folds"], E5C["seed"], E4["n_boot"]
    p0, p1, p2 = crossfit_proba(y, Z, f, s), crossfit_proba(y, np.c_[Z, score], f, s), crossfit_proba(y, score[:, None], f, s)
    resid = crossfit_residual(y, score, Z, f, s)
    out = [("M0", *auroc_ci(y, p0, b, s), CI), ("M1", *auroc_ci(y, p1, b, s), CI), ("M2", *auroc_ci(y, p2, b, s), CI),
           ("M1-M0", *delta_auroc_ci(y, p1, p0, b, s), CI), ("M1-M2", *delta_auroc_ci(y, p1, p2, b, s), CI),
           ("residual", *auroc_ci(y, resid, b, s), CI)]
    gains = [roc_auc_score(y, crossfit_proba(y, np.c_[Z, m], f, s)) - roc_auc_score(y, p0) for m in members]
    resids = [roc_auc_score(y, crossfit_residual(y, m, Z, f, s)) for m in members]
    out += [("M1-M0 members", np.median(gains), min(gains), max(gains), f"median, min-max over {len(members)} members"),
            ("residual members", np.median(resids), min(resids), max(resids), f"median, min-max over {len(members)} members")]
    return out


def sc1(meta):
    rows = []
    for c in TEST_COHORTS:
        m = meta[meta.cohort == c]
        y = m.y.to_numpy()
        S = scores(c, y)
        for src in [s for s in SC1_SOURCES if s in S]:
            score, members = S[src]
            for cov, cols in COVARIATES.items():
                Z = m[cols].to_numpy(float)
                base = dict(source=src, covariates=cov, n=len(y), n_pos=int(y.sum()))
                rows += [row("SC-1", c, metric, v, lo, hi, interval, **base)
                         for metric, v, lo, hi, interval in sc1_one(y, Z, score, members)]
                for t in stratified_auroc(y, score, m[cols[0]].to_numpy(float), E5C["n_bins"], E4["n_boot"], E5C["seed"]):
                    rows.append(row("SC-1", c, f"auroc in {cols[0]} bin", t["auroc"], t["lo"], t["hi"], CI, bin=t["bin"],
                                    strata=f"{t['lo_strata']:.0f}-{t['hi_strata']:.0f}", **{**base, "n": t["n"], "n_pos": t["n_pos"]}))
            print(f"SC-1 {NAMES[c]:10s} {src}", flush=True)
    return rows


def slope_per_doubling(tokens, auroc):
    """Least-squares slope of AUROC on log2(tokens): the AUROC change per doubling of the tokens fed."""
    return float(np.polyfit(np.log2(tokens), auroc, 1)[0])


def sc2(e4, paired_delta):
    """`e4` = results/e4.csv; paired_delta(condition, cohort, base_condition) = e4_report's paired bootstrap on E4's preds."""
    rows = []
    for c in TEST_COHORTS:
        d = e4[e4.test_set == NAMES[c]]
        for kind, ks in (("top", E4["top_k"]), ("rand", E4["random_k"])):
            tokens, aurocs = [], []
            for k in ks:
                g = d[d.condition == f"{kind}{k}"]
                tokens.append(g.mean_tokens.mean())
                aurocs.append(g.auroc.mean())
                rows.append(row("SC-2", c, f"{kind}-K auroc", g.auroc.mean(), g.auroc.min(), g.auroc.max(),
                                "min-max over repeats" if len(g) > 1 else "", K=k, n=len(g), tokens=g.mean_tokens.mean()))
            rows.append(row("SC-2", c, f"{kind}-K slope per doubling", slope_per_doubling(tokens, aurocs), K=f"{ks[0]}-{ks[-1]}"))
        for k in E4["random_k"]:
            r = paired_delta(f"rand{k}", c, base_condition=f"top{k}")
            rows.append(row("SC-2", c, "random-K minus top-K", r["delta"], r["lo"], r["hi"], CI, K=k))
    return rows


def sc3(meta):
    b, s = E4["n_boot"], E5C["seed"]
    rows = []
    for c in TEST_COHORTS:
        m = meta[meta.cohort == c]
        S = scores(c, m.y.to_numpy())
        for other in ("rf_ra", "rf_pa"):
            rows.append(row("SC-3", c, f"rho ensemble vs {other}", *spearman_ci(S["ensemble"][0], S[other][0], b, s), CI))
        for src in ("ensemble", "rf_ra"):
            pairs = [spearmanr(u, v).statistic for u, v in combinations(S[src][1], 2)]
            rows.append(row("SC-3", c, f"rho within {src}", np.median(pairs), min(pairs), max(pairs),
                            f"median, min-max over {len(pairs)} pairs of members"))
        for src, (score, _) in S.items():
            for v in ("richness", "log10_depth"):
                rows.append(row("SC-3", c, f"rho with {v}", *spearman_ci(score, m[v].to_numpy(float), b, s), CI, source=src))
    return rows


def merge_csv(rows, sections):
    """Write rows to results/e5c.csv, replacing earlier rows of the sections just run."""
    new = pd.DataFrame(rows, columns=COLUMNS).astype(str).replace("nan", "")
    if CSV_PATH.exists():
        old = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
        new = pd.concat([old[~old.section.isin(sections)], new])
    new.to_csv(CSV_PATH, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", nargs="+", type=int, default=[0, 1, 2, 3], choices=[0, 1, 2, 3])
    args = ap.parse_args()
    t0 = time.time()
    meta = load_meta()
    rows = []
    if 0 in args.sc:
        rows += sc0(meta)
    if 1 in args.sc:
        rows += sc1(meta)
    if 2 in args.sc:
        from e4_report import paired_delta
        rows += sc2(pd.read_csv(RESULTS / "e4.csv"), paired_delta)
    if 3 in args.sc:
        rows += sc3(meta)
    merge_csv(rows, [f"SC-{i}" for i in args.sc])
    print(f"{len(rows)} rows -> {CSV_PATH}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
