"""Step 8 outputs: figures and results/phase0_summary.md from results/phase0_rf.csv and results/preds/.

  figures/fig1_encoding_comparison.png    spec RF (primary): AUROC by encoding, one panel per test set
  figures/fig1b_encoding_comparison_unweighted.png   same for the unweighted-RF sensitivity arm
  figures/fig2_shift_severity.png         AUROC vs ASV overlap with AGP (Pope's BLAST values), one line per encoding
  results/phase0_summary.md               medians + CIs, in-domain-minus-cross-study gaps, paired deltas vs ra

Per encoding x test set: the point is the median AUROC over seeds, the error bar the median over seeds of the
per-seed bootstrap 95% CI, small dots the individual seeds. Paired deltas vs `ra` average each encoding's predicted
probabilities over seeds, then bootstrap the AUROC difference on shared resamples (1,000).
"""
import json
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from _paths import FIGURES, PROCESSED, RESULTS  # noqa: E402

ENCODINGS = ["ra", "pa", "rank_within", "clr", "rank_raw"]
COLORS = dict(zip(ENCODINGS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))  # validated, light mode
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PANELS = [  # (test_set, title, Pope No-DR reference)
    ("HMP2", "HMP2 (cross-study)", 0.657),
    ("Halfvarson", "Halfvarson (cross-study)", 0.530),
    ("AGP-IBD-holdout", "AGP-IBD shipped holdout\n(no leakage; batch-shifted)", 0.717),
    ("AGP-IBD-5fold", "AGP-IBD 5-fold CV\n(patient leakage)", 0.717),
]
POPE_OVERLAP = {"Halfvarson": 0.34, "HMP2": 0.51, "AGP-IBD-holdout": 1.0}
PRED_FILE = {"HMP2": "cross_hmp2", "Halfvarson": "cross_hv", "AGP-IBD-holdout": "holdout", "AGP-IBD-5fold": "cv"}
N_BOOT = 1000


def load_results():
    df = pd.read_csv(RESULTS / "phase0_rf.csv")
    df = df[(df.status == "OK") & (df.data_variant == "canonical") & df.encoding.isin(ENCODINGS)].copy()
    df["rf"] = np.where(df.rf_params.str.contains('"class_weight": null'), "unweighted", "spec")
    return df.drop_duplicates(["encoding", "test_set", "seed", "rf"], keep="last")


def summarise(df):
    g = df.groupby(["rf", "encoding", "test_set"])
    return g.agg(auroc=("auroc", "median"), lo=("auroc_ci_lo", "median"), hi=("auroc_ci_hi", "median"),
                 auprc=("auprc", "median"), n_seeds=("seed", "nunique"),
                 ties=("mean_ties_per_sample", "mean")).reset_index()


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_encodings(df, summ, rf, path, title, ncols=None):
    panels = [p for p in PANELS if ((df.rf == rf) & (df.test_set == p[0])).any()]
    ncols = ncols or len(panels)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 4.4 * nrows), sharey=True, facecolor=SURFACE,
                             squeeze=False)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for r in range(nrows):
        axes[r, 0].set_ylabel("AUROC", color=INK)
    axes = axes.ravel()
    rng = np.random.default_rng(0)
    for ax, (test, ptitle, pope) in zip(axes, panels):
        style(ax)
        ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.axhline(pope, color=INK, lw=1, ls=(0, (1, 2)), zorder=1)
        for i, e in enumerate(ENCODINGS):
            s = summ[(summ.rf == rf) & (summ.encoding == e) & (summ.test_set == test)]
            pts = df[(df.rf == rf) & (df.encoding == e) & (df.test_set == test)].auroc.values
            if s.empty:
                continue
            s = s.iloc[0]
            ax.plot([i, i], [s.lo, s.hi], color=COLORS[e], lw=2, solid_capstyle="round", zorder=2)
            ax.scatter(i + rng.uniform(-0.07, 0.07, len(pts)) + 0.22, pts, s=14, color=COLORS[e], alpha=0.55,
                       edgecolor="none", zorder=3)
            ax.scatter([i], [s.auroc], s=64, color=COLORS[e], edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.set_xticks(range(len(ENCODINGS)), ENCODINGS, rotation=30, ha="right", color=INK)
        ax.set_xlim(-0.6, len(ENCODINGS) - 0.3)
        ax.set_title(f"{ptitle}\nPope No-DR (dotted): {pope:.3f}", fontsize=10, color=INK, loc="left")
    axes[0].set_ylim(0.2, 1.0)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    note = ("Large dot = median over 5 seeds; bar = median per-seed bootstrap 95% CI (1,000 resamples; "
            "ignores repeated samples per patient); small dots = seeds. Dashed = 0.5; dotted = Pope et al. 2025 No-DR.")
    fig.text(0.01, 0.01, textwrap.fill(note, width=int(38 * ncols)), fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.07 / nrows, 1, 1 - 0.05 / nrows))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_shift(summ, path):
    fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor=SURFACE)
    style(ax)
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    order = ["Halfvarson", "HMP2", "AGP-IBD-holdout"]
    for k, e in enumerate(ENCODINGS):
        dx = (k - 2) * 0.008
        s = summ[(summ.rf == "spec") & (summ.encoding == e)].set_index("test_set")
        xs = [POPE_OVERLAP[t] + dx for t in order if t in s.index]
        ys = [s.loc[t, "auroc"] for t in order if t in s.index]
        lo = [s.loc[t, "lo"] for t in order if t in s.index]
        hi = [s.loc[t, "hi"] for t in order if t in s.index]
        ax.vlines(xs, lo, hi, color=COLORS[e], lw=1.5, alpha=0.6, zorder=2)
        ax.plot(xs, ys, color=COLORS[e], lw=2, zorder=3, label=e)
        ax.scatter(xs, ys, s=56, color=COLORS[e], edgecolor=SURFACE, linewidth=2, zorder=4)
        if "AGP-IBD-5fold" in s.index:
            ax.scatter([1.0 + dx + 0.045], [s.loc["AGP-IBD-5fold", "auroc"]], s=40, facecolor=SURFACE,
                       edgecolor=COLORS[e], linewidth=1.5, zorder=4)
    ax.annotate("hollow = AGP 5-fold CV\n(patient leakage, upper bound)", xy=(1.045, 0.80), xytext=(0.80, 0.93),
                fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xticks([0.34, 0.51, 1.0], ["Halfvarson\n0.34", "HMP2\n0.51", "AGP-IBD shipped holdout\n1.0 (batch-shifted)"],
                  color=INK)
    ax.set_xlim(0.25, 1.12)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("ASV overlap with AGP (Pope et al. 2025 BLAST: >=99% identity, e-value 1e-20)", color=INK2,
                  fontsize=9)
    ax.set_ylabel("AUROC (median over 5 seeds)", color=INK)
    leg = ax.legend(title="encoding", fontsize=8.5, title_fontsize=8.5, frameon=False, loc="upper left",
                    bbox_to_anchor=(1.01, 1.0), labelcolor=INK)
    leg.get_title().set_color(INK2)
    ax.set_title("AUROC vs distribution shift, spec RF (train AGP-IBD)", fontsize=11.5, color=INK, loc="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def paired_deltas(rf):
    """AUROC(encoding) - AUROC(ra) on seed-averaged predictions, paired bootstrap CI."""
    tag = "canonical" if rf == "spec" else "canonical_unweighted"
    out = []
    for test, key in PRED_FILE.items():
        preds = {}
        for e in ENCODINGS:
            files = sorted((RESULTS / "preds").glob(f"{e}_{tag}_{key}_seed*.npz"))
            if files:
                arrs = [np.load(f) for f in files]
                preds[e] = (arrs[0]["y"], np.mean([a["p"] for a in arrs], axis=0), len(files))
        if "ra" not in preds:
            continue
        y, p_ra, _ = preds["ra"]
        rng = np.random.default_rng(0)
        boot = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
        boot = [b for b in boot if y[b].min() != y[b].max()]
        for e, (ye, pe, n) in preds.items():
            if e == "ra":
                continue
            d = roc_auc_score(y, pe) - roc_auc_score(y, p_ra)
            ds = [roc_auc_score(y[b], pe[b]) - roc_auc_score(y[b], p_ra[b]) for b in boot]
            lo, hi = np.percentile(ds, [2.5, 97.5])
            out.append(dict(test_set=test, encoding=e, delta=d, lo=lo, hi=hi, n_seeds=n))
    return pd.DataFrame(out)


def fmt(r):
    return f"{r.auroc:.3f} [{r.lo:.3f}, {r.hi:.3f}]" + ("" if r.n_seeds == 5 else f" (n={r.n_seeds})")


def summary_md(df, summ):
    tests = [p[0] for p in PANELS]
    L = [f"# Phase 0 summary — abundance encodings under cross-study evaluation",
         "", f"_Generated {datetime.now():%Y-%m-%d %H:%M} by `src/report.py` from `results/phase0_rf.csv`._", "",
         "Train on AGP-IBD, test on HMP2 and Halfvarson with no fine-tuning; in-domain references on AGP-IBD. "
         "Cell = **median AUROC over seeds [median per-seed bootstrap 95% CI]**. Bootstrap CIs resample samples "
         "independently and ignore repeated samples per patient (HMP2/HV are longitudinal), so they are too narrow.",
         "", "Pope et al. 2025 No-DR reference: HMP2 0.657 · Halfvarson 0.530 · AGP-IBD 0.717 (patient-blocked).", ""]
    for rf, head in [("spec", "Primary: spec RF (500 trees, sqrt, leaf 3, class_weight='balanced')"),
                     ("unweighted", "Sensitivity: unweighted RF (identical but class_weight=None, as DeepMicro)")]:
        s = summ[summ.rf == rf]
        if s.empty:
            continue
        present = [t for t in tests if t in set(s.test_set)]
        L += [f"## {head}", "", "| encoding | " + " | ".join(present) + " |", "|---" * (len(present) + 1) + "|"]
        for e in ENCODINGS:
            cells = []
            for t in present:
                r = s[(s.encoding == e) & (s.test_set == t)]
                cells.append(fmt(r.iloc[0]) if not r.empty else "—")
            L.append(f"| `{e}` | " + " | ".join(cells) + " |")
        L.append("")
        # gaps
        L += ["**In-domain minus cross-study gap** (median AUROC; positive = worse under shift)", "",
              "| encoding | holdout − HMP2 | holdout − HV | 5-fold − HMP2 | 5-fold − HV |", "|---|---|---|---|---|"]
        for e in ENCODINGS:
            m = s[s.encoding == e].set_index("test_set").auroc
            g = lambda a, b: f"{m[a] - m[b]:+.3f}" if a in m and b in m else "—"
            L.append(f"| `{e}` | {g('AGP-IBD-holdout', 'HMP2')} | {g('AGP-IBD-holdout', 'Halfvarson')} | "
                     f"{g('AGP-IBD-5fold', 'HMP2')} | {g('AGP-IBD-5fold', 'Halfvarson')} |")
        L.append("")
        d = paired_deltas(rf)
        if not d.empty:
            L += ["**Paired difference vs `ra`** (AUROC of seed-averaged predictions, encoding − ra, paired bootstrap "
                  "95% CI; CI excluding 0 marked *)", "", "| encoding | " + " | ".join(present) + " |",
                  "|---" * (len(present) + 1) + "|"]
            for e in ENCODINGS[1:]:
                cells = []
                for t in present:
                    r = d[(d.encoding == e) & (d.test_set == t)]
                    if r.empty:
                        cells.append("—")
                        continue
                    r = r.iloc[0]
                    star = "*" if (r.lo > 0 or r.hi < 0) else ""
                    cells.append(f"{r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]{star}")
                L.append(f"| `{e}` | " + " | ".join(cells) + " |")
            L.append("")
    ties = summ[(summ.rf == "spec") & summ.encoding.isin(["rank_within", "rank_raw"])]
    if not ties.empty:
        L += ["## Rank ties (mean tied non-zero entries per sample, spec RF runs)", "",
              "| encoding | " + " | ".join(tests) + " |", "|---" * (len(tests) + 1) + "|"]
        for e in ["rank_within", "rank_raw"]:
            cells = []
            for t in tests:
                r = ties[(ties.encoding == e) & (ties.test_set == t)]
                cells.append(f"{r.iloc[0].ties:.1f}" if not r.empty else "—")
            L.append(f"| `{e}` | " + " | ".join(cells) + " |")
        L.append("")
    m = json.loads((PROCESSED / "manifest.json").read_text())["cohorts"]
    L += ["## Caveats", "",
          "- **In-domain 5-fold CV is leakage-inflated** (S2: 32% of IBD test samples have an IBD nearest training "
          "neighbour vs 5% base rate). Patient IDs are unavailable.",
          "- **The shipped holdout is leakage-free but batch-shifted**: an RF separates Pope's IBD_train from IBD_test "
          "with AUROC 0.935 (5-fold, `ra`); IBD_test has min depth 5,115 vs 17 and higher richness (median 251 vs 216), "
          "i.e. it was processed differently. There is therefore **no clean in-domain estimate** in this data: 5-fold CV "
          "is an upper bound, the shipped holdout behaves like another shifted cohort, and the 'gap' columns must be read "
          "with that in mind.",
          "- **Halfvarson is fragile**: with `ra`, AUROC moves from 0.818 (balanced RF) to 0.340 (unweighted RF), "
          "seed 0. Compare encodings within an RF arm, never across.",
          f"- ASV overlap on Fig 2 is Pope et al.'s BLAST figure. Our ID-match overlap is "
          f"{m['hmp2']['asv_overlap_frac_idmatch']:.3f} (HMP2) / {m['hv']['asv_overlap_frac_idmatch']:.3f} (HV), "
          "near 1 by construction because the cohorts were pre-mapped to the AGP vocabulary.",
          "- One vocabulary (26,726 IDs), one classifier per arm, no per-encoding tuning; all encoding statistics "
          "fitted on training data only.", ""]
    (RESULTS / "phase0_summary.md").write_text("\n".join(L))


def main():
    FIGURES.mkdir(exist_ok=True)
    df = load_results()
    summ = summarise(df)
    fig_encodings(df, summ, "spec", FIGURES / "fig1_encoding_comparison.png",
                  "AUROC by abundance encoding: spec RF, trained on AGP-IBD")
    fig_encodings(df, summ, "spec", FIGURES / "fig1_encoding_comparison_2x2.png",  # portrait layout for the PDF report
                  "AUROC by abundance encoding: spec RF, trained on AGP-IBD", ncols=2)
    if (df.rf == "unweighted").any():
        fig_encodings(df, summ, "unweighted", FIGURES / "fig1b_encoding_comparison_unweighted.png",
                      "Sensitivity: unweighted RF (class_weight=None), trained on AGP-IBD")
    fig_shift(summ, FIGURES / "fig2_shift_severity.png")
    summary_md(df, summ)
    print(summ[summ.rf == "spec"].to_string(index=False))


if __name__ == "__main__":
    main()
