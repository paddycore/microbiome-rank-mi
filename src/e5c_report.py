"""E5c outputs: Figure 6, a score-vs-richness figure and results/e5c_summary.md from results/e5c.csv and e5c_meta.csv.

  figures/fig_e5c_tokens.png   Figure 6 (SC-2): AUROC against the mean number of tokens fed, top-K vs random-K, per
                               test cohort. At one K both feed the same tokens, so the vertical gap is the effect of
                               which taxa and the slope the effect of how many.
  figures/fig_e5c_scores.png   P(IBD) against richness per test cohort, for Pope's ensemble and the RF (ra).
  results/e5c_summary.md       every number, SC-0 to SC-3

Run it at any stop point: sections not computed yet are skipped. Wording: everything but SC-2 is correlational, so the
summary says "explained by", never "uses".
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
from e5c_signal import COVARIATES, CSV_PATH, META_CSV, NAMES, SC0_VARIABLES, SC1_SOURCES, TEST_COHORTS, scores

E4, E5C = pm.CFG["e4"], pm.CFG["e5c"]
SURFACE, INK, INK2, GRID = p0.SURFACE, p0.INK, p0.INK2, p0.GRID
KIND_COLORS = {"top": "#2a78d6", "rand": "#eb6834"}                                    # validated pair, light mode
KIND_LABELS = {"top": "top-K (most abundant first)", "rand": "random-K"}
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}
CLASS_LABELS = {0: "control", 1: "IBD"}
SOURCE_LABELS = {"ensemble": "Pope's ensemble", "rf_ra": "RF, relative abundance", "rf_pa": "RF, presence/absence",
                 "e5b_epoch1": "E5b heads, 1 epoch", "e5b_epoch50": "E5b heads, 50 epochs"}
VARIABLE_LABELS = {"richness": "richness (all taxa)", "n_taxa": "taxa the model sees (max 512)",
                   "richness_ids": "richness (distinct vocab IDs)", "log10_depth": "log10 read depth"}


def load():
    if not CSV_PATH.exists():
        return None
    d = pd.read_csv(CSV_PATH, keep_default_na=False)
    for col in ("value", "lo", "hi", "tokens", "K", "bin"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def ci(r, digits=3):
    if pd.isna(r.value):
        return "–"
    return f"{r.value:.{digits}f} [{r.lo:.{digits}f}, {r.hi:.{digits}f}]" if pd.notna(r.lo) else f"{r.value:.{digits}f}"


def direction(r):
    """Which class has the higher value, only when the CI excludes 0.5."""
    if r.lo > 0.5:
        return "higher in IBD"
    if r.hi < 0.5:
        return "higher in controls"
    return "no clear direction"


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *["| " + " | ".join(map(str, r)) + " |" for r in rows]]


def spread(values, gap):
    """Label heights pushed at least `gap` apart, keeping their order (labels of lines that end close together)."""
    order = np.argsort(values)
    out = np.array(values, dtype=float)[order]
    for _ in range(len(out)):
        for i in range(1, len(out)):
            if out[i] - out[i - 1] < gap:
                mid = (out[i] + out[i - 1]) / 2
                out[i - 1], out[i] = mid - gap / 2, mid + gap / 2
    return dict(zip(order, out))


def fig_tokens(d, path):
    s = d[d.section == "SC-2"]
    fig, axes = plt.subplots(1, len(TEST_COHORTS), figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    for ax, c in zip(axes, TEST_COHORTS):
        p0.style(ax)
        ends = {}
        for kind, color in KIND_COLORS.items():
            r = s[(s.cohort == NAMES[c]) & (s.metric == f"{kind}-K auroc")].sort_values("tokens")
            if kind == "rand":
                ax.fill_between(r.tokens, r.lo, r.hi, color=color, alpha=0.15, lw=0)
            ax.plot(r.tokens, r.value, color=color, lw=2, zorder=2)
            ax.scatter(r.tokens, r.value, s=40, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ends[kind] = (r.tokens.iloc[-1], r.value.iloc[-1])
        x_end = max(x for x, _ in ends.values())
        placed = spread([v for _, v in ends.values()], 0.03)
        for i, kind in enumerate(ends):
            ax.annotate(KIND_LABELS[kind], xy=(x_end, placed[i]), xytext=(10, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK2, annotation_clip=False, bbox=dict(fc=SURFACE, ec="none", pad=0.5))
        slopes = {k: s[(s.cohort == NAMES[c]) & (s.metric == f"{k}-K slope per doubling")].value for k in KIND_COLORS}
        ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xscale("log", base=2)
        here = s[s.cohort == NAMES[c]].tokens
        ticks = [t for t in (16, 32, 64, 128, 256, 512) if here.min() * 0.9 <= t <= here.max() * 1.1]
        ax.set_xticks(ticks, [str(t) for t in ticks], color=INK)
        ax.minorticks_off()
        ax.set_xlabel("mean tokens fed per sample (log scale)", color=INK2, fontsize=8.5)
        ax.set_title(NAMES[c], loc="left", fontsize=10.5, color=INK, fontweight="bold", pad=17)
        ax.text(0, 1.015, "AUROC change per doubling: " + ", ".join(f"{k}-K {v.iloc[0]:+.3f}" for k, v in slopes.items() if len(v)),
                transform=ax.transAxes, fontsize=8.5, color=INK2)
    axes[0].set_ylabel("AUROC, Pope's ensemble", color=INK, fontsize=9)
    fig.legend(handles=[Line2D([], [], color=c, lw=2, marker="o", ms=6, label=KIND_LABELS[k]) for k, c in KIND_COLORS.items()],
               loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    note = ("E4d at test time: each sample keeps K taxa, either its K most abundant or K at random (band = min-max over "
            f"{E4['random_k_repeats']} random draws). At one K both feed the same number of tokens, so the vertical gap is "
            "the effect of which taxa and the slope the effect of how many. Dashed = chance.")
    fig.text(0.01, 0.005, textwrap.fill(note, 170), fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_scores(meta, d, path):
    sources = ["ensemble", "rf_ra"]
    fig, axes = plt.subplots(len(TEST_COHORTS), len(sources), figsize=(10, 8), facecolor=SURFACE)
    for row_axes, c in zip(axes, TEST_COHORTS):
        m = meta[meta.cohort == c]
        S = scores(c, m.y.to_numpy())
        for ax, src in zip(row_axes, sources):
            p0.style(ax)
            for cls in (0, 1):
                k = m.y.to_numpy() == cls
                ax.scatter(m.richness[k], S[src][0][k], s=18, color=CLASS_COLORS[cls], alpha=0.7, edgecolor=SURFACE,
                           linewidth=0.6, zorder=3 if cls == 0 else 2)
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_title(f"{NAMES[c]}, {SOURCE_LABELS[src]}", loc="left", fontsize=10, color=INK, fontweight="bold", pad=17)
            r = d[(d.section == "SC-3") & (d.cohort == NAMES[c]) & (d.metric == "rho with richness") & (d.source == src)]
            if len(r):
                ax.text(0, 1.015, f"Spearman ρ with richness {ci(r.iloc[0], 2)}", transform=ax.transAxes, fontsize=8.5, color=INK2)
            ax.set_xlabel("richness (taxa with reads, log scale)", color=INK2, fontsize=8.5)
            ax.set_ylabel("P(IBD)", color=INK, fontsize=9)
    fig.legend(handles=[Line2D([], [], color=c, lw=0, marker="o", ms=6, label=CLASS_LABELS[k]) for k, c in CLASS_COLORS.items()],
               loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    note = ("Each dot is a sample of the test cohort. Richness counts every taxon with reads in Pope's full list (no 512 cap). "
            "Correlational: a score that tracks richness is explained by it, which does not show the model uses it.")
    fig.text(0.01, 0.005, textwrap.fill(note, 160), fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def sc0_section(d):
    s = d[d.section == "SC-0"]
    if s.empty:
        return []
    L = ["\n## SC-0: richness and read depth alone (HC1)\n",
         "AUROC of each variable for IBD, bootstrap 95% CI. Below 0.5 = higher in controls. Direction is named only when the "
         "CI excludes 0.5.\n"]
    rows = []
    for v in SC0_VARIABLES:
        cells = []
        for c in pm.CFG["e5"]["cohorts"]:
            r = s[(s.source == v) & (s.cohort == NAMES[c])]
            cells.append(f"{ci(r.iloc[0])}, {direction(r.iloc[0])}" if len(r) else "–")
        rows.append([VARIABLE_LABELS[v], *cells])
    return L + table(["Variable", *[NAMES[c] for c in pm.CFG["e5"]["cohorts"]]], rows)


SC1_METRICS = [("M0", "M0: richness + depth"), ("M2", "M2: score alone"), ("M1", "M1: both"),
               ("M1-M0", "Δ M1 − M0 (score over simple signals)"), ("M1-M2", "Δ M1 − M2 (simple signals over score)"),
               ("residual", "AUROC of residual score")]


def sc1_section(d):
    s = d[d.section == "SC-1"]
    if s.empty:
        return []
    L = ["\n## SC-1: what the score adds beyond richness and depth (HC2)\n",
         f"Logistic models cross-fitted over {E5C['n_folds']} folds; Δ = paired bootstrap 95% CI on the out-of-fold "
         "predictions. Members: median [min–max] over the models averaged into each source (10 heads, 5 RF seeds, 5 E5b "
         "seeds). Covariates `n_taxa` = richness as the model sees it (capped at 512).\n"]
    for c in TEST_COHORTS:
        for cov in COVARIATES:
            sub = s[(s.cohort == NAMES[c]) & (s.covariates == cov)]
            if sub.empty:
                continue
            rows = []
            for src in [x for x in SC1_SOURCES if x in set(sub.source)]:
                g = sub[sub.source == src].set_index("metric")
                cells = [ci(g.loc[m]) for m, _ in SC1_METRICS]
                cells += [ci(g.loc["M1-M0 members"]), ci(g.loc["residual members"])]
                rows.append([SOURCE_LABELS[src], *cells])
            L += [f"\n**{NAMES[c]}, covariates: {cov} + log10 depth**\n",
                  *table(["Source", *[t for _, t in SC1_METRICS], "Δ M1 − M0, members", "residual, members"], rows)]
            bins = sub[sub.metric.str.startswith("auroc in")]
            if len(bins):
                rows = [[SOURCE_LABELS[src], *[f"{ci(r)} (n {r.n}, IBD {r.n_pos})" for _, r in g.sort_values("bin").iterrows()]]
                        for src, g in bins.groupby("source", sort=False)]
                strata = bins.drop_duplicates("bin").sort_values("bin").strata.tolist()
                L += [f"\n_AUROC of the score inside {cov} tertiles ({', '.join(strata)})_\n",
                      *table(["Source", *[f"tertile {i + 1}" for i in range(len(strata))]], rows)]
    return L


def sc2_section(d):
    s = d[d.section == "SC-2"]
    if s.empty:
        return []
    L = ["\n## SC-2: which taxa vs how many (HC4)\n",
         "Pope's ensemble with K taxa kept per sample (E4d). Random-K: mean [min–max] over draws. Gap: paired bootstrap "
         "95% CI (E4's preds).\n"]
    for c in TEST_COHORTS:
        g = s[s.cohort == NAMES[c]]
        rows = []
        for k in sorted(g.K.dropna().unique()):
            top = g[(g.metric == "top-K auroc") & (g.K == k)]
            rand = g[(g.metric == "rand-K auroc") & (g.K == k)]
            gap = g[(g.metric == "random-K minus top-K") & (g.K == k)]
            rows.append([int(k), f"{top.tokens.iloc[0]:.0f}" if len(top) else "–", ci(top.iloc[0]) if len(top) else "–",
                         (f"{rand.value.iloc[0]:.3f} [{rand.lo.iloc[0]:.3f}–{rand.hi.iloc[0]:.3f}]" if len(rand) else "–"),
                         ci(gap.iloc[0]) if len(gap) else "–"])
        slopes = ", ".join(f"{k}: {g[g.metric == f'{k}-K slope per doubling'].value.iloc[0]:+.3f}" for k in KIND_COLORS
                           if (g.metric == f"{k}-K slope per doubling").any())
        L += [f"\n**{NAMES[c]}** (AUROC change per doubling of tokens: {slopes})\n",
              *table(["K", "Mean tokens", "Top-K", "Random-K", "Random − top"], rows)]
    return L


def sc3_section(d):
    s = d[d.section == "SC-3"]
    if s.empty:
        return []
    L = ["\n## SC-3: do the RF and the ensemble rank samples alike (HC3)\n",
         "Spearman ρ with bootstrap 95% CI. Yardstick: ρ between two members of the same source, median [min–max] over "
         "pairs. A ρ(ensemble, RF) inside the yardstick range means the two cannot be told apart by how they rank.\n"]
    rows = []
    for c in TEST_COHORTS:
        g = s[s.cohort == NAMES[c]].set_index("metric")
        within = lambda m: (f"{g.loc[m].value:.3f} [{g.loc[m].lo:.3f}–{g.loc[m].hi:.3f}]" if m in g.index else "–")
        rows.append([NAMES[c], ci(g.loc["rho ensemble vs rf_ra"]), ci(g.loc["rho ensemble vs rf_pa"]),
                     within("rho within ensemble"), within("rho within rf_ra")])
    L += table(["Cohort", "ensemble vs RF ra", "ensemble vs RF pa", "within ensemble", "within RF ra"], rows)
    rows = []
    for src in SOURCE_LABELS:
        cells = []
        for c in TEST_COHORTS:
            for v in ("richness", "log10_depth"):
                r = s[(s.cohort == NAMES[c]) & (s.source == src) & (s.metric == f"rho with {v}")]
                cells.append(ci(r.iloc[0]) if len(r) else "–")
        rows.append([SOURCE_LABELS[src], *cells])
    L += ["\n**ρ of each score with richness and depth**\n",
          *table(["Source", *[f"{NAMES[c]}: {v}" for c in TEST_COHORTS for v in ("richness", "log10 depth")]], rows)]
    return L


def summary_md(d, path):
    L = [f"# E5c summary\n\n_Generated {datetime.now():%Y-%m-%d %H:%M} by `src/e5c_report.py` from `results/e5c.csv`._\n",
         "Everything except SC-2 is correlational: a score that tracks richness is *explained by* it, which does not show "
         "the model *uses* it. CIs ignore repeated samples per patient (no subject IDs ship with the data), so they are "
         "too narrow. Cohorts are never pooled.\n"]
    L += sc0_section(d) + sc1_section(d) + sc2_section(d) + sc3_section(d)
    path.write_text("\n".join(L) + "\n")


def main():
    d = load()
    if d is None:
        print(f"no {CSV_PATH} yet: run src/e5c_signal.py first")
        return
    if (d.section == "SC-2").any():
        fig_tokens(d, FIGURES / "fig_e5c_tokens.png")
    if META_CSV.exists():
        fig_scores(pd.read_csv(META_CSV), d, FIGURES / "fig_e5c_scores.png")
    summary_md(d, RESULTS / "e5c_summary.md")
    print((RESULTS / "e5c_summary.md").read_text())


if __name__ == "__main__":
    main()
