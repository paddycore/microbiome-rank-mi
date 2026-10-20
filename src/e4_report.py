"""E4 outputs: figures and results/e4_summary.md from results/e4.csv, results/preds/e4_*.npz, results/e4c.csv.

  figures/fig_e4_order_budget.png   A: ΔAUROC vs Pope's order per perturbation (E4a), B: AUROC vs K (E4d)
  figures/fig_e4c_baselines.png     order-free baselines (E4c) next to the transformer and Phase 0 RF
  results/e4_summary.md             every number in the figures, plus the S4-0 gate

ΔAUROC is paired: on each of n_boot shared resamples of the test samples, the condition's AUROC (averaged over its
repeats, e.g. 20 random permutations) minus the AUROC of Pope's order. CIs ignore repeated samples per patient
(no subject IDs ship with the data), so they are too narrow, as in Phase 0.
"""
import re
import textwrap
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import pope_model as pm
import report as p0
from _paths import FIGURES, RESULTS

E4 = pm.CFG["e4"]
COHORTS = {pm.CFG["cohorts"][c]["name"]: c for c in E4["test_cohorts"]}       # "HMP2" -> "hmp2"
COLORS = {"HMP2": "#2a78d6", "Halfvarson": "#eb6834"}                          # validated pair, light mode
PERTURB = [("ties", "shuffle within tied counts\n(noise floor)"), ("shuffle", "shuffle all taxa"),
           ("reverse", "reverse (least abundant first)"), ("pope_dropout", "Pope's test-time dropout 10%")]
SURFACE, INK, INK2, GRID = p0.SURFACE, p0.INK, p0.INK2, p0.GRID


def cohort_legend(fig):
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=c, markeredgecolor=SURFACE, label=t)
               for t, c in COLORS.items()]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK)


def ensemble_probs(condition, cohort):
    """{repeat: (y, mean P(IBD) over the 10 models)} from the saved per-model probabilities."""
    out = {}
    for f in sorted((RESULTS / "preds").glob(f"e4_{condition}_r*_{cohort}.npz")):
        z = np.load(f)
        out[int(re.search(r"_r(\d+)_[a-z0-9]+\.npz$", f.name).group(1))] = (z["y"], z["probs"].mean(0))
    return out


def paired_delta(condition, cohort):
    base = ensemble_probs("sorted", cohort)
    reps = ensemble_probs(condition, cohort)
    if not base or not reps:
        return None
    y, p0_ = base[0]
    ps = [p for _, p in reps.values()]
    rng = np.random.default_rng(0)
    boot = [b for b in (rng.integers(0, len(y), len(y)) for _ in range(E4["n_boot"])) if y[b].min() != y[b].max()]
    d = np.mean([roc_auc_score(y, p) for p in ps]) - roc_auc_score(y, p0_)
    ds = [np.mean([roc_auc_score(y[b], p[b]) for p in ps]) - roc_auc_score(y[b], p0_[b]) for b in boot]
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(delta=d, lo=lo, hi=hi, n_repeats=len(ps))


def summarise(df):
    rows = []
    for (cond, test), g in df.groupby(["condition", "test_set"]):
        r = dict(condition=cond, test_set=test, n_repeats=len(g), auroc=g.auroc.mean(), auroc_min=g.auroc.min(),
                 auroc_max=g.auroc.max(), ci_lo=g.auroc_ci_lo.mean(), ci_hi=g.auroc_ci_hi.mean(),
                 tokens=g.mean_tokens.mean())
        if cond != "sorted":
            r.update(paired_delta(cond, COHORTS[test]) or {})
        rows.append(r)
    return pd.DataFrame(rows)


def verdict(r):
    if pd.isna(r.get("lo")):
        return ""
    if r["lo"] > 0:
        return "above baseline (CI excludes 0)"
    if r["hi"] < 0:
        return "below baseline (CI excludes 0)"
    return "CI includes 0"


def fig_order_budget(s, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.15, 1]))
    p0.style(a)
    a.grid(axis="y", visible=False)
    a.grid(axis="x", color=GRID, linewidth=0.8)
    a.axvline(0, color=INK2, lw=1, zorder=1)
    conds = [(c, lab) for c, lab in PERTURB if (s.condition == c).any()]
    for i, (c, _) in enumerate(conds):
        for k, test in enumerate(COLORS):
            r = s[(s.condition == c) & (s.test_set == test)]
            if r.empty or pd.isna(r.iloc[0].get("delta")):
                continue
            r = r.iloc[0]
            yy = i + (0.14 if k == 0 else -0.14)
            a.plot([r.lo, r.hi], [yy, yy], color=COLORS[test], lw=2, solid_capstyle="round", zorder=2)
            a.scatter([r.delta], [yy], s=64, color=COLORS[test], edgecolor=SURFACE, linewidth=2, zorder=3)
    a.set_yticks(range(len(conds)), [lab for _, lab in conds], color=INK, fontsize=9)
    a.invert_yaxis()
    a.set_xlabel("ΔAUROC vs Pope's order (paired bootstrap 95% CI)", color=INK2, fontsize=9)
    a.set_title("A  Perturb token order at test time (E4a)", loc="left", fontsize=10.5, color=INK, fontweight="bold")

    p0.style(b)
    for test, c in COHORTS.items():
        r = s[s.condition.str.match(r"top\d+$") & (s.test_set == test)].copy()
        if r.empty:
            continue
        r["k"] = r.condition.str[3:].astype(int)
        r = r.sort_values("k")
        b.fill_between(r.k, r.ci_lo, r.ci_hi, color=COLORS[test], alpha=0.10, lw=0)
        b.plot(r.k, r.auroc, color=COLORS[test], lw=2, zorder=2)
        b.scatter(r.k, r.auroc, s=56, color=COLORS[test], edgecolor=SURFACE, linewidth=2, zorder=3)
        n_tok = s[(s.condition == "sorted") & (s.test_set == test)].tokens.iloc[0]
        b.axvline(n_tok, color=COLORS[test], lw=1, ls=(0, (1, 2)), zorder=1)
    b.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    b.set_xscale("log", base=2)
    b.set_xticks(E4["top_k"], [str(k) for k in E4["top_k"]], color=INK)
    b.set_xlabel("K = most abundant taxa kept (log scale)", color=INK2, fontsize=9)
    b.set_ylabel("AUROC (ensemble)", color=INK)
    b.set_title("B  Token budget (E4d)", loc="left", fontsize=10.5, color=INK, fontweight="bold")

    note = ("Pope et al.'s 10-model fine-tuned ensemble, trained on AGP-IBD, tested on HMP2 (n=197) and Halfvarson "
            "(n=564). A: shuffles averaged over 20 random permutations. B: band = bootstrap 95% CI; dotted vertical = "
            "mean taxa per sample in that cohort (K above it changes few samples). CIs ignore repeated samples per "
            "patient.")
    fig.text(0.01, 0.01, textwrap.fill(note, 170), fontsize=7.5, color=INK2)
    cohort_legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def baselines(s):
    """Rows for the E4c comparison: transformer, bag-of-embeddings (if run), Phase 0 RF."""
    out = []
    for test in COLORS:
        for cond, lab in [("sorted", "Transformer, Pope's order"), ("shuffle", "Transformer, shuffled (20 perms)")]:
            r = s[(s.condition == cond) & (s.test_set == test)]
            if not r.empty:
                r = r.iloc[0]
                out.append(dict(model=lab, test_set=test, auroc=r.auroc, lo=r.ci_lo, hi=r.ci_hi))
        if (RESULTS / "e4c.csv").exists():
            for _, r in pd.read_csv(RESULTS / "e4c.csv").query("test_set == @test").iterrows():
                cw = "balanced" if r.class_weight == "balanced" else "unweighted"
                out.append(dict(model=f"Bag of embeddings, {r.bag}, LR {cw}", test_set=test, auroc=r.auroc,
                                lo=r.auroc_ci_lo, hi=r.auroc_ci_hi))
        summ = p0.summarise(p0.load_results())
        for enc, lab in [("ra", "RF relative abundance (Phase 0)"), ("pa", "RF presence/absence (Phase 0)")]:
            r = summ[(summ.rf == "spec") & (summ.encoding == enc) & (summ.test_set == test)].iloc[0]
            out.append(dict(model=lab, test_set=test, auroc=r.auroc, lo=r.lo, hi=r.hi))
    return pd.DataFrame(out)


def fig_baselines(bl, path):
    models = list(dict.fromkeys(bl.model))
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(models) + 1.6), facecolor=SURFACE)
    p0.style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.axvline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    for i, m in enumerate(models):
        for k, test in enumerate(COLORS):
            r = bl[(bl.model == m) & (bl.test_set == test)]
            if r.empty:
                continue
            r = r.iloc[0]
            yy = i + (0.15 if k == 0 else -0.15)
            ax.plot([r.lo, r.hi], [yy, yy], color=COLORS[test], lw=2, solid_capstyle="round", zorder=2)
            ax.scatter([r.auroc], [yy], s=56, color=COLORS[test], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_yticks(range(len(models)), models, color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Cross-study AUROC (bootstrap 95% CI)", color=INK2, fontsize=9)
    ax.set_title("Order-free baselines vs the transformer (E4c)", loc="left", fontsize=10.5, color=INK,
                 fontweight="bold")
    cohort_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def summary_md(s, bl, path):
    f3 = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    sd = lambda v: "" if pd.isna(v) else f"{v:+.3f}"
    L = [f"# E4 summary\n\n_Generated {datetime.now():%Y-%m-%d %H:%M} by `src/e4_report.py` from `results/e4.csv`, "
         "`results/preds/e4_*.npz`, `results/e4c.csv`._\n",
         "Model: Pope et al.'s 10-model fine-tuned ensemble (`ensemble.zip`), trained on AGP-IBD; test cohorts are "
         "cross-study. Ensemble score = mean P(IBD) over the 10 models (as in Pope's `begin.py`). "
         "ΔAUROC = condition minus Pope's order, paired bootstrap on shared resamples "
         f"({E4['n_boot']}); for repeated conditions the condition's AUROC is the mean over repeats. "
         "CIs ignore repeated samples per patient.\n",
         "## S4-0 reproduction gate\n", "| Cohort | Pope et al. | Ours, Pope's order [95% CI] | Ours, Pope's 10% test dropout (mean of 10) |",
         "|---|---|---|---|"]
    for test, c in COHORTS.items():
        r = s[(s.condition == "sorted") & (s.test_set == test)].iloc[0]
        d = s[(s.condition == "pope_dropout") & (s.test_set == test)]
        L.append(f"| {test} | {E4['pope_auroc'][c]:.3f} | {r.auroc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}] | "
                 f"{f3(d.auroc.iloc[0]) if not d.empty else ''} |")
    L += ["\n## E4a: token order at test time\n",
          "| Condition | Test set | Repeats | AUROC (mean) | Range over repeats | ΔAUROC vs Pope's order [95% CI] | Reading |",
          "|---|---|---|---|---|---|---|"]
    for c, lab in [("sorted", "Pope's order")] + PERTURB:
        for test in COLORS:
            r = s[(s.condition == c) & (s.test_set == test)]
            if r.empty:
                continue
            r = r.iloc[0]
            delta = "" if c == "sorted" else f"{sd(r.get('delta'))} [{sd(r.get('lo'))}, {sd(r.get('hi'))}]"
            rng_ = f"{r.auroc_min:.3f}–{r.auroc_max:.3f}" if r.n_repeats > 1 else ""
            L.append(f"| {lab.replace(chr(10), ' ')} | {test} | {r.n_repeats} | {r.auroc:.3f} | {rng_} | {delta} | "
                     f"{'' if c == 'sorted' else verdict(r)} |")
    L += ["\n## E4d: token budget (top-K most abundant taxa)\n",
          "| K | Test set | Mean taxa kept | AUROC [95% CI] | ΔAUROC vs all taxa [95% CI] | Reading |", "|---|---|---|---|---|---|"]
    for k in E4["top_k"]:
        for test in COLORS:
            r = s[(s.condition == f"top{k}") & (s.test_set == test)]
            if r.empty:
                continue
            r = r.iloc[0]
            L.append(f"| {k} | {test} | {r.tokens:.0f} | {r.auroc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}] | "
                     f"{sd(r.get('delta'))} [{sd(r.get('lo'))}, {sd(r.get('hi'))}] | {verdict(r)} |")
    L += ["\n## E4c: order-free baselines (absolute AUROC, not paired)\n", "| Model | HMP2 | Halfvarson |", "|---|---|---|"]
    for m in dict.fromkeys(bl.model):
        cells = []
        for test in COLORS:
            r = bl[(bl.model == m) & (bl.test_set == test)]
            cells.append("" if r.empty else f"{r.auroc.iloc[0]:.3f} [{r.lo.iloc[0]:.3f}, {r.hi.iloc[0]:.3f}]")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    if not (RESULTS / "e4c.csv").exists():
        L.append("\n_Bag-of-embeddings rows appear after `python src/e4c_bag.py` is run._")
    L.append("\nPhase 0 RF rows: spec RF (`class_weight='balanced'`), median over 5 seeds, full 1,897-column input "
             "(not the top-512 input the transformer sees).")
    path.write_text("\n".join(L) + "\n")


def main():
    df = pd.read_csv(RESULTS / "e4.csv")
    s = summarise(df)
    bl = baselines(s)
    fig_order_budget(s, FIGURES / "fig_e4_order_budget.png")
    fig_baselines(bl, FIGURES / "fig_e4c_baselines.png")
    summary_md(s, bl, RESULTS / "e4_summary.md")
    print((RESULTS / "e4_summary.md").read_text())


if __name__ == "__main__":
    main()
