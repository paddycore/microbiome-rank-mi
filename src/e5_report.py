"""E5 outputs: Figure 3 and results/e5_summary.md from results/e5.csv.

  figures/fig_e5_probes.png   per layer (0 = encoder input, 1-5 = encoder layers), mean pooling:
                              A cohort-probe selectivity, raw (solid) and after partialling out read depth + taxa
                                count (dashed); grey line = the same probe on [log10 depth, n_taxa] alone
                              B disease-probe selectivity inside AGP-IBD (random CV, patient leakage: upper bound)
                              C cross-study AUROC of a fresh linear head trained on AGP-IBD (HMP2, Halfvarson)
                              D how well the confounders can be read back out (ridge CV R^2)
  results/e5_summary.md       every number, both poolings, and the probe-capacity sweep (C)

Ensemble = Pope's 10 fine-tuned models: line = mean over the 10, band = their min-max. pre120 = the pretrained
checkpoint they were fine-tuned from; pre0 = untrained (random-embedding control).
"""
import textwrap
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import pope_model as pm
import report as p0
from _paths import FIGURES, RESULTS

E5 = pm.CFG["e5"]
SURFACE, INK, INK2, GRID = p0.SURFACE, p0.INK, p0.INK2, p0.GRID
GROUPS = {"ensemble": "#2a78d6", "pre120": "#eb6834", "pre0": "#1baf7a"}   # validated categorical order (Phase 0)
LABELS = {"ensemble": "Pope ensemble (fine-tuned, n=10)", "pre120": "pretrained, before fine-tuning",
          "pre0": "untrained (control)"}


def load():
    d = pd.read_csv(RESULTS / "e5.csv", keep_default_na=False)
    base = d[d.model == "confounders_only"].iloc[0]
    d = d[d.model != "confounders_only"].copy()
    for col in ("layer", "C", "score"):
        d[col] = pd.to_numeric(d[col])
    d["selectivity"] = pd.to_numeric(d.selectivity.replace("", np.nan))
    d["control"] = pd.to_numeric(d.control.replace("", np.nan))
    d["group"] = np.where(d.model.str.startswith("ens"), "ensemble", d.model)
    return d, base


def summarise(d):
    """mean / min / max over models within a group (only the ensemble has more than one)."""
    g = d.groupby(["group", "pooling", "C", "task", "layer"])
    return g.agg(score=("score", "mean"), score_min=("score", "min"), score_max=("score", "max"),
                 sel=("selectivity", "mean"), sel_min=("selectivity", "min"), sel_max=("selectivity", "max"),
                 control=("control", "mean"), n_models=("model", "nunique")).reset_index()


def panel(ax, s, task, value, title, ylabel, ls="-", band=True, label=True):
    for group, color in GROUPS.items():
        r = s[(s.group == group) & (s.task == task)].sort_values("layer")
        if r.empty:
            continue
        lo, hi = (r[f"{value}_min"], r[f"{value}_max"]) if f"{value}_min" in r else (None, None)
        if band and group == "ensemble" and lo is not None:
            ax.fill_between(r.layer, lo, hi, color=color, alpha=0.12, lw=0)
        ax.plot(r.layer, r[value], color=color, lw=2, ls=ls, zorder=2)
        ax.scatter(r.layer, r[value], s=44 if ls == "-" else 30, zorder=3,
                   color=color if ls == "-" else SURFACE, edgecolor=SURFACE if ls == "-" else color, linewidth=2)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.set_xticks(range(6), ["input"] + [str(i) for i in range(1, 6)], color=INK)
    ax.set_xlabel("hidden state (0 = encoder input, 1-5 = encoder layers)", color=INK2, fontsize=8.5)


def fig_probes(d, base, path):
    s = summarise(d[(d.pooling == "mean") & (d.C == E5["C_main"])])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), facecolor=SURFACE)
    (a, b), (c, e) = axes
    for ax in axes.ravel():
        p0.style(ax)
    panel(a, s, "cohort", "sel", "A  Which cohort? (selectivity)", "balanced accuracy - control")
    panel(a, s, "cohort_resid", "sel", None, "balanced accuracy - control", ls=(0, (4, 2)), band=False)
    a.axhline(float(base.selectivity), color=INK2, lw=1.2, ls=(0, (1, 2)), zorder=1)
    a.text(5.1, float(base.selectivity), "depth + taxa\ncount alone", fontsize=7.5, color=INK2, va="center")
    a.set_xlim(-0.3, 5.9)
    panel(b, s, "disease_agp", "sel", "B  IBD inside AGP-IBD (selectivity, leaky CV)", "balanced accuracy - control")
    for task, ls in (("transfer_hmp2", "-"), ("transfer_hv", (0, (4, 2)))):
        panel(c, s, task, "score", None, "AUROC", ls=ls)
    c.set_title("C  Fresh linear head on AGP-IBD, cross-study", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    c.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    for task, ls in (("depth_r2", "-"), ("ntaxa_r2", (0, (4, 2)))):
        panel(e, s, task, "score", None, "CV R²", ls=ls)
    e.set_title("D  Confounders read out of the hidden state", loc="left", fontsize=10.5, color=INK, fontweight="bold")

    handles = [Line2D([], [], color=col, lw=2, marker="o", ms=6, label=LABELS[g]) for g, col in GROUPS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    keys = {a: ("raw", "depth + taxa count partialled out"), c: ("HMP2", "Halfvarson"), e: ("log10 read depth", "taxa count")}
    for ax, (solid, dashed) in keys.items():
        ax.legend(handles=[Line2D([], [], color=INK2, lw=2, label=solid),
                           Line2D([], [], color=INK2, lw=2, ls=(0, (4, 2)), marker="o", ms=5, mfc=SURFACE, label=dashed)],
                  frameon=False, fontsize=8, labelcolor=INK, loc="best")
    note = (f"Linear probes (logistic regression, C={E5['C_main']}, class-balanced, {E5['n_folds']}-fold stratified CV) on "
            "mean-pooled hidden states. Selectivity = balanced accuracy minus the same probe on permuted labels "
            f"(Hewitt & Liang 2019). Cohort probe: AGP-IBD subsampled to {E5['agp_subsample']} + all HMP2 and "
            "Halfvarson. Band = min-max over the 10 ensemble models. Samples from the same patient can sit in "
            "different folds (no patient IDs), which inflates within-cohort scores.")
    fig.text(0.01, 0.005, textwrap.fill(note, 175), fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def summary_md(d, base, path):
    s = summarise(d)
    f = lambda v: f"{v:.3f}"
    rng = lambda r, k: f"{r[k]:.3f}" + (f" [{r[k + '_min']:.3f}–{r[k + '_max']:.3f}]" if r.n_models > 1 else "")
    L = [f"# E5 summary\n\n_Generated {datetime.now():%Y-%m-%d %H:%M} by `src/e5_report.py` from `results/e5.csv`._\n",
         f"Probes: logistic regression, class-balanced, {E5['n_folds']}-fold stratified CV, balanced accuracy. "
         "Selectivity = score − score on permuted labels. Ensemble values: mean over the 10 models [min–max]. "
         "Layer 0 = encoder input; its [CLS] vector is identical for every sample, so [CLS] probes there sit at chance.\n",
         f"Baseline, cohort probe on [log10 depth, n_taxa] only: balanced accuracy {float(base.score):.3f}, "
         f"control {float(base.control):.3f}, selectivity **{float(base.selectivity):.3f}**.\n"]
    tasks = [("cohort", "sel", "Cohort selectivity"), ("cohort_resid", "sel", "Cohort selectivity, depth + taxa count partialled out"),
             ("disease_agp", "sel", "IBD selectivity inside AGP-IBD (leaky CV)"),
             ("transfer_hmp2", "score", "Fresh linear head, AUROC on HMP2"), ("transfer_hv", "score", "Fresh linear head, AUROC on Halfvarson"),
             ("depth_r2", "score", "log10 read depth, CV R²"), ("ntaxa_r2", "score", "taxa count, CV R²")]
    for pooling in ("mean", "cls"):
        L.append(f"\n## Pooling: {pooling}, C = {E5['C_main']}\n")
        for task, key, title in tasks:
            L += [f"\n**{title}**\n", "| Model | " + " | ".join(["input"] + [f"L{i}" for i in range(1, 6)]) + " |",
                  "|---|" + "---|" * 6]
            for group in GROUPS:
                r = s[(s.group == group) & (s.task == task) & (s.pooling == pooling) & (s.C == E5["C_main"])].sort_values("layer")
                if not r.empty:
                    L.append(f"| {LABELS[group]} | " + " | ".join(rng(x, key) for _, x in r.iterrows()) + " |")
    L.append("\n## Probe capacity (mean pooling): cohort selectivity by C\n")
    L += ["| Model | C | " + " | ".join(["input"] + [f"L{i}" for i in range(1, 6)]) + " |", "|---|---|" + "---|" * 6]
    for group in ("ensemble", "pre120", "pre0"):
        for C in E5["C"]:
            sub = d[(d.pooling == "mean") & (d.task == "cohort") & (d.C == C) &
                    ((d.model == "ens1") if group == "ensemble" else (d.model == group))].sort_values("layer")
            if not sub.empty:
                L.append(f"| {'ens1' if group == 'ensemble' else group} | {C} | " + " | ".join(f(v) for v in sub.selectivity) + " |")
    path.write_text("\n".join(L) + "\n")


def main():
    d, base = load()
    fig_probes(d, base, FIGURES / "fig_e5_probes.png")
    summary_md(d, base, RESULTS / "e5_summary.md")
    print((RESULTS / "e5_summary.md").read_text())


if __name__ == "__main__":
    main()
