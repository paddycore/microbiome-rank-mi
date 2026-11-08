"""E10: does the alignment of the disease directions of two cohorts predict cross-cohort transfer? (notebook/plan_e10.md)

For every cached feature space (Pope's 12 models x layers x poolings from E5, E7's abundance poolings, E9's three
Micro16S bodies), standardised with the AGP 80% statistics:
  disease direction  Δμ = mean(IBD) - mean(control); AGP's from its 80% training rows, a test cohort's from all its rows
                     (this uses the test labels: a diagnostic, never a way to pick a model)
  alignment          cos(Δμ_AGP, Δμ_test), with a null from 200 shuffles of the test labels
  transfer           AUROC on the test cohort of (a) the projection onto Δμ_AGP (the class-mean-difference classifier,
                     the C -> 0 limit of L2 logistic regression), (b) LR at C = 1e-4, (c) LR at C = 1, all fitted on AGP 80%
  separation         |Δμ_test|: how far apart the test classes are in this space, whatever the direction

Usage: python src/e10_alignment.py     (-> results/e10.csv, results/e10_summary.md, figures/fig_e10_alignment.png)
"""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import e5b_head as h
import pope_model as pm
import report as p0
from _paths import FIGURES, RESULTS
from e5_states import state_path

COHORTS = pm.CFG["e5"]["cohorts"]
FEATURES = pm.path("features_dir")
CSV, SUMMARY = RESULTS / "e10.csv", RESULTS / "e10_summary.md"
N_NULL = 200
FAMILY_COLORS = {"Pope fine-tuned": "#2a78d6", "Pope pretrained / untrained": "#eb6834", "Micro16S": "#1baf7a"}  # validated all-pairs


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def analyse(feats, seed=0):
    """Rows for one feature space: alignment, null, separation and the three transfer AUROCs per eval set."""
    rows_tr, sets = h.eval_sets(feats, "80")
    X, y = feats["agp_ibd"]
    mu, sd = X[rows_tr].mean(0), X[rows_tr].std(0) + 1e-8
    Xs, ytr = (X[rows_tr] - mu) / sd, y[rows_tr]
    w = Xs[ytr == 1].mean(0) - Xs[ytr == 0].mean(0)
    fits = {c: LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(Xs, ytr) for c in (1e-4, 1.0)}
    rng = np.random.default_rng(seed)
    out = []
    for key, (A, yk) in sets.items():
        As = (A - mu) / sd
        d = As[yk == 1].mean(0) - As[yk == 0].mean(0)
        null = []
        for _ in range(N_NULL):
            p = rng.permutation(yk)
            null.append(cosine(w, As[p == 1].mean(0) - As[p == 0].mean(0)))
        out.append(dict(test_set=h.SET_NAMES[key], cos=cosine(w, d), null_lo=np.percentile(null, 2.5),
                        null_hi=np.percentile(null, 97.5), separation=float(np.linalg.norm(d)),
                        auroc_meandiff=roc_auc_score(yk, As @ w),
                        auroc_lr_strong=roc_auc_score(yk, fits[1e-4].decision_function(As)),
                        auroc_lr_weak=roc_auc_score(yk, fits[1.0].decision_function(As))))
    return out


def pope_model_rows(model):
    family = "Pope fine-tuned" if model.startswith("ens") else "Pope pretrained / untrained"
    e5 = {c: np.load(state_path(model, c)) for c in COHORTS}
    e7 = {c: np.load(FEATURES / f"e7_{model}_{c}.npz") for c in COHORTS}
    spaces = [(f"{pooling} L{layer}", {c: (e5[c][pooling][:, layer].astype(np.float32), e5[c]["y"].astype(int)) for c in COHORTS})
              for pooling in ("mean", "cls") for layer in range(6) if not (pooling == "cls" and layer == 0)]
    spaces += [(f"{pooling} L5", {c: (e7[c][pooling], e7[c]["y"].astype(int)) for c in COHORTS}) for pooling in ("abund", "abund_shuffled")]
    return [dict(family=family, model=model, space=name, **r) for name, feats in spaces for r in analyse(feats)]


def micro16s_rows(body):
    feats = {c: (np.load(FEATURES / f"e9_{body}_{c}.npz")["X"], np.load(FEATURES / f"e9_{body}_{c}.npz")["y"].astype(int)) for c in COHORTS}
    return [dict(family="Micro16S", model=body, space=f"micro16s {body}", **r) for r in analyse(feats)]


def figure(d, path):
    tests = ["HMP2", "Halfvarson"]
    measures = [("auroc_meandiff", "class-mean-difference classifier"), ("auroc_lr_weak", "logistic regression, C = 1")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True, facecolor=p0.SURFACE)
    for i, (col, label) in enumerate(measures):
        for j, t in enumerate(tests):
            ax = axes[i, j]
            p0.style(ax)
            s = d[d.test_set == t]
            for fam, color in FAMILY_COLORS.items():
                f = s[s.family == fam]
                ax.scatter(f.cos, f[col], s=26 if fam != "Micro16S" else 70, color=color, edgecolor=p0.SURFACE, linewidth=0.8,
                           alpha=0.85, zorder=3 if fam == "Micro16S" else 2)
            ax.axhline(0.5, color=p0.INK2, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.axvline(0, color=p0.GRID, lw=1, zorder=1)
            rho = spearmanr(s.cos, s[col]).statistic
            ax.set_title(f"{t}: {label}", loc="left", fontsize=10, color=p0.INK, fontweight="bold", pad=17)
            ax.text(0, 1.015, f"Spearman ρ = {rho:.2f} over {len(s)} feature spaces", transform=ax.transAxes, fontsize=8.5, color=p0.INK2)
            if i == 1:
                ax.set_xlabel("cos(AGP disease direction, test-cohort disease direction)", color=p0.INK2, fontsize=8.5)
            if j == 0:
                ax.set_ylabel("cross-cohort AUROC", color=p0.INK, fontsize=9)
    fig.legend(handles=[Line2D([], [], color=c, lw=0, marker="o", ms=7, label=k) for k, c in FAMILY_COLORS.items()],
               loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=p0.INK)
    note = ("Each dot is one frozen feature space (model x layer x pooling), standardised with AGP-IBD 80% statistics. Disease "
            "direction = mean(IBD) - mean(control). The test-cohort direction uses the test labels, so alignment is a diagnostic, "
            "not a model-selection tool. Dashed = chance.")
    fig.text(0.01, 0.005, textwrap.fill(note, 165), fontsize=7.5, color=p0.INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=200, facecolor=p0.SURFACE)
    plt.close(fig)


def summary(d):
    L = ["# E10 summary: alignment of disease directions vs transfer\n",
         f"{d.space.nunique()} space types, {len(d) // 3} feature spaces; cos with a null from {N_NULL} test-label shuffles.\n",
         "## Spearman ρ between alignment and transfer, over feature spaces\n",
         "| Eval set | class-mean difference | LR C = 1e-4 | LR C = 1 | separation vs LR C = 1 |", "|---|---|---|---|---|"]
    for t, s in d.groupby("test_set"):
        L.append(f"| {t} | {spearmanr(s.cos, s.auroc_meandiff).statistic:.2f} | {spearmanr(s.cos, s.auroc_lr_strong).statistic:.2f} | "
                 f"{spearmanr(s.cos, s.auroc_lr_weak).statistic:.2f} | {spearmanr(s.separation, s.auroc_lr_weak).statistic:.2f} |")
    L += ["\n## By family: median cos [min–max], share of spaces above the null, median AUROC (mean diff / LR C = 1)\n",
          "| Family | Eval set | cos | above null | AUROC mean diff | AUROC LR C=1 |", "|---|---|---|---|---|---|"]
    for (fam, t), s in d.groupby(["family", "test_set"]):
        L.append(f"| {fam} | {t} | {s.cos.median():.2f} [{s.cos.min():.2f}–{s.cos.max():.2f}] | {(s.cos > s.null_hi).mean():.0%} | "
                 f"{s.auroc_meandiff.median():.3f} | {s.auroc_lr_weak.median():.3f} |")
    m = d[d.family == "Micro16S"][["model", "test_set", "cos", "null_lo", "null_hi", "auroc_meandiff", "auroc_lr_strong", "auroc_lr_weak"]]
    L += ["\n## Micro16S spaces\n", "```", m.round(3).to_string(index=False), "```"]
    p = d[(d.model == "ens1") & d.space.isin(["cls L5", "mean L5", "abund L5"])][["space", "test_set", "cos", "null_hi", "auroc_meandiff", "auroc_lr_weak"]]
    L += ["\n## Pope ens1, last layer (for comparison)\n", "```", p.round(3).to_string(index=False), "```"]
    SUMMARY.write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    jobs = [delayed(pope_model_rows)(m) for m in h.ALL_MODELS] + [delayed(micro16s_rows)(b) for b in ("pre", "rand", "bag")]
    rows = [r for part in Parallel(n_jobs=4, verbose=0)(jobs) for r in part]
    d = pd.DataFrame(rows)
    d.round(4).to_csv(CSV, index=False)
    figure(d[d.test_set != h.SET_NAMES[h.HOLDOUT]], FIGURES / "fig_e10_alignment.png")
    summary(d)


if __name__ == "__main__":
    main()
