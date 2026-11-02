"""E5b outputs: figures and results/e5b_summary.md from results/e5b_gate.csv, e5b.csv, e5b_lr.csv, preds/e5b_*.npz.

  figures/fig_e5b_epochs.png     Figure 5, AUROC vs epochs of head training (Pope's recipe, head on 80% of AGP-IBD):
                                 A ensemble of 10 heads on Pope's bodies, B single heads on those bodies,
                                 C single heads on pre120, D single heads on pre0.
                                 If HB1 holds, the hold-out line rises while the cross-study lines fall.
  figures/fig_e5b_lr.png         the same four panels against C, with (solid) and without (dashed) a scaler (HB2, HB3)
  figures/fig_e5b_ablation.png   each ablation against Pope's recipe, ensemble level (HB4)
  results/e5b_summary.md         every number in the figures, the SB-0 table, paired differences

Run it at any stop point: whatever is not there yet is skipped.

Band = min-max: over seeds for an ensemble, over heads for single heads. ΔAUROC is paired: on each of n_boot shared
resamples of the samples, on seed-averaged predictions. CIs ignore repeated samples per patient (no subject IDs ship
with the data), so they are too narrow, as in Phase 0. The hold-out shares patients with the training rows, so in-domain
numbers are an upper bound.
"""
import textwrap
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

import pope_model as pm
import report as p0
from _paths import FIGURES, RESULTS
from e5b_head import CONFIGS, GATE_CSV, HEAD_CSV, HOLDOUT, LR_CSV, MODELS, PRED_DIR, SET_NAMES, gate_passes

E4, E5B = pm.CFG["e4"], pm.CFG["e5b"]
SURFACE, INK, INK2, GRID = p0.SURFACE, p0.INK, p0.INK2, p0.GRID
COLORS = {"HMP2": "#2a78d6", "Halfvarson": "#eb6834", SET_NAMES[HOLDOUT]: "#1baf7a"}     # validated all-pairs, light mode
CONFIG_COLORS = dict(zip(CONFIGS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))  # validated, lines
PANELS = {"ensemble": "A  Ensemble of 10 heads, Pope's bodies", "members": "B  Single heads, Pope's bodies",
          "pre120": "C  Single heads, pretrained body (pre120)", "pre0": "D  Single heads, untrained body (pre0)"}
DASHED = (0, (4, 2))
EPOCH_TICKS = [0, 1, 2, 5, 10, 20, 50]


def load(path):
    if not path.exists():
        return None
    d = pd.read_csv(path, keep_default_na=False)
    for col in ("auroc", "auroc_ci_lo", "auroc_ci_hi"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["panel"] = np.where(d.model == "ensemble", "ensemble", np.where(d.model.str.startswith("ens"), "members", d.model))
    return d


def curve(d, x):
    """mean / min / max AUROC at each x."""
    return d.groupby(x).auroc.agg(["mean", "min", "max"]).reset_index().sort_values(x)


def own_heads(gate):
    """{panel: {eval set: AUROC}} of Pope's own heads on the cached features, {} before SB-0 has run."""
    if gate is None:
        return {}
    own = gate[gate.config == "pope_own_head"]
    return {p: own[own.panel == p].groupby("test_set").auroc.mean().to_dict() for p in ("ensemble", "members")}


def draw(ax, r, x, color, ls="-", band=True, markers=True, lw=2):
    if band:
        ax.fill_between(r[x], r["min"], r["max"], color=color, alpha=0.12, lw=0)
    ax.plot(r[x], r["mean"], color=color, lw=lw, ls=ls, zorder=2)
    if markers:
        ax.scatter(r[x], r["mean"], s=36 if ls == "-" else 26, zorder=3, color=color if ls == "-" else SURFACE,
                   edgecolor=SURFACE if ls == "-" else color, linewidth=1.5)


def spread(values, gap):
    """Push label heights apart by at least `gap`, keeping their order and their mean."""
    order = np.argsort(values)
    out = np.array(values, dtype=float)[order]
    for _ in range(len(out)):
        for i in range(1, len(out)):
            if out[i] - out[i - 1] < gap:
                mid = (out[i] + out[i - 1]) / 2
                out[i - 1], out[i] = mid - gap / 2, mid + gap / 2
    return dict(zip(order, out))


def end_labels(ax, ends, x):
    """Direct labels at the right-hand end of the lines, in text ink: the legend's colours are never the only key."""
    placed = spread(list(ends.values()), 0.035)
    for i, name in enumerate(ends):
        ax.annotate(name, xy=(x, placed[i]), xytext=(9, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK2, annotation_clip=False, bbox=dict(fc=SURFACE, ec="none", pad=0.5))


def reference_lines(ax, refs):
    for test_set, v in refs.items():
        ax.axhline(v, color=COLORS[test_set], lw=1, ls=(0, (1, 2)), zorder=1)


def own_subtitle(refs):
    return "dotted: Pope's own heads, " + ", ".join(f"{k} {v:.3f}" for k, v in refs.items()) if refs else ""


def finish(ax, title, subtitle):
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold", pad=17)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2)


def epoch_axis(ax):
    ax.set_xscale("symlog", linthresh=1, linscale=0.5)
    ax.set_xlim(-0.25, 60)
    ax.set_xticks(EPOCH_TICKS, [str(e) for e in EPOCH_TICKS], color=INK)
    ax.minorticks_off()
    ax.set_xlabel("epochs of head training (0 = untrained, log scale)", color=INK2, fontsize=8.5)


def save(fig, path, note, wrap=165):
    fig.text(0.01, 0.005, textwrap.fill(note, wrap), fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_epochs(d, own, path):
    s = d[d.config == "pope"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, (panel, title) in zip(axes.ravel(), PANELS.items()):
        p0.style(ax)
        ax.axvline(1, color=GRID, lw=1, zorder=1)
        ends = {}
        for test_set, color in COLORS.items():
            r = curve(s[(s.panel == panel) & (s.test_set == test_set)], "epoch")
            if not r.empty:
                draw(ax, r, "epoch", color)
                ends[test_set] = r["mean"].iloc[-1]
        reference_lines(ax, own.get(panel, {}))
        finish(ax, title, own_subtitle(own.get(panel, {})))
        end_labels(ax, ends, s.epoch.max())
        epoch_axis(ax)
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC", color=INK, fontsize=9)
    fig.legend(handles=[Line2D([], [], color=c, lw=2, marker="o", ms=6, label=k) for k, c in COLORS.items()],
               loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    train = s[s.epoch == 1].iloc[0]
    save(fig, path, f"Heads with Pope's recipe on frozen [CLS] features, {len(E5B['seeds'])} seeds per body, trained on "
         f"{train.n_train} AGP-IBD rows, so 1 epoch = {train.steps} steps (Pope: 268, on all of AGP-IBD). Vertical line = "
         "1 epoch, where Pope's shipped models stopped. Band = min-max over seeds (A) or over heads (B-D). Dashed = chance. "
         "The hold-out shares patients with the training rows: an upper bound.")


def fig_lr(lr, own, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, (panel, title) in zip(axes.ravel(), PANELS.items()):
        p0.style(ax)
        for test_set, color in COLORS.items():
            for scale, ls in ((True, "-"), (False, DASHED)):
                r = lr[(lr.panel == panel) & (lr.test_set == test_set) & (lr.scale.astype(str) == str(scale))]
                if not r.empty:
                    draw(ax, curve(r, "C"), "C", color, ls=ls, band=False)
        reference_lines(ax, own.get(panel, {}))
        finish(ax, title, own_subtitle(own.get(panel, {})))
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xlabel("logistic regression C (weaker regularisation to the right)", color=INK2, fontsize=8.5)
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC", color=INK, fontsize=9)
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=6, label=k) for k, c in COLORS.items()]
    handles += [Line2D([], [], color=INK2, lw=2, label="with StandardScaler"),
                Line2D([], [], color=INK2, lw=2, ls=DASHED, marker="o", ms=5, mfc=SURFACE, label="raw features")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9, labelcolor=INK)
    save(fig, path, f"Logistic regression (class-balanced) on the same frozen [CLS] features, trained on {lr.n_train.iloc[0]} "
         "AGP-IBD rows. A: mean P(IBD) over the 10 bodies. B: mean AUROC over the 10 bodies. Dashed grey = chance.")


def fig_ablation(d, own, path):
    names = [c for c in CONFIGS if c in set(d.config)]
    if len(names) < 2:
        return
    s = d[d.panel == "ensemble"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True, facecolor=SURFACE)
    for ax, test_set in zip(axes, COLORS):
        p0.style(ax)
        ax.axvline(1, color=GRID, lw=1, zorder=1)
        for name in names:
            r = curve(s[(s.config == name) & (s.test_set == test_set)], "epoch")
            if not r.empty:
                draw(ax, r, "epoch", CONFIG_COLORS[name], band=False, markers=False, lw=2.5 if name == "pope" else 1.5)
        v = own.get("ensemble", {}).get(test_set)
        if v is not None:
            ax.axhline(v, color=INK, lw=1, ls=(0, (1, 2)), zorder=1)
        finish(ax, test_set, f"dotted: Pope's own heads, {v:.3f}" if v is not None else "")
        epoch_axis(ax)
    axes[0].set_ylabel("AUROC, ensemble of 10 heads", color=INK, fontsize=9)
    fig.legend(handles=[Line2D([], [], color=CONFIG_COLORS[n], lw=2.5 if n == "pope" else 1.5, label=n) for n in names],
               loc="upper center", ncol=len(names), frameon=False, fontsize=9, labelcolor=INK)
    save(fig, path, "Each config changes one thing from Pope's recipe (pope). Mean over seeds. ce and mse move at different "
         "speeds under one lr, so read the whole curve, not the single epoch. Vertical line = 1 epoch.", wrap=170)


def seed_averaged(name):
    """{"epochs": list, eval set: (y, P(IBD) of shape (epochs, n))}, or None if that config has no ensemble predictions."""
    path = PRED_DIR / f"e5b_{name}_80.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return {"epochs": z["epochs"].tolist(), **{k: (z[f"{k}_y"], z[f"{k}_p"].mean(0)) for k in SET_NAMES}}


def paired_delta(y, a, b):
    """AUROC(a) - AUROC(b) with a percentile CI from n_boot shared resamples."""
    rng = np.random.default_rng(0)
    boot = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(E4["n_boot"])) if y[i].min() != y[i].max()]
    lo, hi = np.percentile([roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in boot], [2.5, 97.5])
    return roc_auc_score(y, a) - roc_auc_score(y, b), lo, hi


def fmt_delta(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def fmt_range(r):
    return f"{r['mean']:.3f} [{r['min']:.3f}–{r['max']:.3f}]"


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *["| " + " | ".join(map(str, r)) + " |" for r in rows]]


def gate_section(gate):
    if gate is None:
        return []
    L = ["\n## SB-0: Pope's recipe against Pope's own heads\n",
         f"Passes when the mean over seeds of the fresh ensemble is within {E4['gate_tolerance']} of Pope's own heads on the "
         "same cached features, per cohort, heads on all of AGP-IBD for 1 epoch. Ensemble: mean [min–max] over seeds. "
         "Member: mean ± sd over heads. Epoch 0 = untrained heads.\n"]
    rows = []
    for key, name in SET_NAMES.items():
        own = gate[(gate.config == "pope_own_head") & (gate.test_set == name)]
        if len(own):
            ens, mem = own[own.panel == "ensemble"].iloc[0], own[own.panel == "members"].auroc
            rows.append([name, f"Pope's own (paper {E4['pope_auroc'][key]:.3f})", "", f"{ens.auroc:.3f} [{ens.auroc_ci_lo:.3f}, "
                         f"{ens.auroc_ci_hi:.3f}]", f"{mem.mean():.3f} ± {mem.std():.3f}", ""])
        for train_on, epoch in (("all", 0), ("all", 1), ("80", 0), ("80", 1)):
            r = gate[(gate.config == "pope") & (gate.train_on == train_on) & (gate.epoch == epoch) & (gate.test_set == name)]
            if r.empty:
                continue
            e, m = r[r.panel == "ensemble"].auroc, r[r.panel == "members"].auroc
            verdict = ("PASS" if gate_passes(e, ens.auroc) else "FAIL") if len(own) and (train_on, epoch) == ("all", 1) else ""
            rows.append([name, f"fresh, {train_on}", epoch, f"{e.mean():.3f} [{e.min():.3f}–{e.max():.3f}]",
                         f"{m.mean():.3f} ± {m.std():.3f}", verdict])
    return L + table(["Eval set", "Heads", "Epoch", "Ensemble", "Member", "Verdict"], rows)


def sweep_section(d):
    if d is None or "pope" not in set(d.config):
        return []
    s = d[d.config == "pope"]
    L = ["\n## SB-1: epoch sweep, Pope's recipe (HB1)\n",
         f"Head trained on {s.n_train.iloc[0]} AGP-IBD rows. Mean [min–max]: over seeds for the ensemble, over heads otherwise.\n"]
    for panel, title in PANELS.items():
        sub = s[s.panel == panel]
        if sub.empty:
            continue
        rows = []
        for epoch, g in sub.groupby("epoch"):
            cells = [fmt_range(curve(g[g.test_set == t], "epoch").iloc[0]) for t in COLORS]
            rows.append([epoch, g.steps.iloc[0], *cells])
        L += [f"\n**{title}**\n", *table(["Epoch", "Steps", *COLORS], rows)]
    preds = seed_averaged("pope")
    if preds:
        L += ["\n**Paired difference from epoch 1** (ensemble, seed-averaged predictions). HB1 predicts hold-out above 0, "
              "cross-study below 0.\n"]
        base = preds["epochs"].index(1)
        rows = [[e, *[fmt_delta(paired_delta(preds[k][0], preds[k][1][preds["epochs"].index(e)], preds[k][1][base]))
                      for k in SET_NAMES]] for e in E5B["ci_epochs"] + [5, 10, 20] if e != 1 and e in preds["epochs"]]
        L += table(["Epoch", *[SET_NAMES[k] for k in SET_NAMES]], sorted(rows, key=lambda r: r[0]))
    return L


def lr_section(lr):
    if lr is None:
        return []
    L = ["\n## SB-1: logistic regression sweep (HB2, HB3)\n",
         f"Trained on {lr.n_train.iloc[0]} AGP-IBD rows. Ensemble: mean P(IBD) over the 10 bodies, with a bootstrap CI. "
         "Single heads: mean [min–max] over the 10 bodies.\n"]
    for panel, title in PANELS.items():
        sub = lr[lr.panel == panel]
        if sub.empty:
            continue
        rows = []
        for (C, scale), g in sub.groupby(["C", "scale"], sort=True):
            if panel == "ensemble":
                cells = [(lambda r: f"{r.auroc:.3f} [{r.auroc_ci_lo:.3f}, {r.auroc_ci_hi:.3f}]")(g[g.test_set == t].iloc[0]) for t in COLORS]
            elif panel == "members":
                cells = [fmt_range(curve(g[g.test_set == t], "C").iloc[0]) for t in COLORS]
            else:
                cells = [f"{g[g.test_set == t].auroc.iloc[0]:.3f}" for t in COLORS]
            rows.append([C, "scaled" if str(scale) == "True" else "raw", *cells])
        L += [f"\n**{title}**\n", *table(["C", "Features", *COLORS], rows)]
    return L


def ablation_section(d):
    names = [] if d is None else [c for c in CONFIGS if c != "pope" and c in set(d.config)]
    base = seed_averaged("pope")
    if not names or base is None:
        return []
    L = ["\n## SB-2: ablations (HB4)\n", "One change each from Pope's recipe. Ensemble level, mean over seeds, and the paired "
         "difference from `pope` at the same epoch (seed-averaged predictions).\n"]
    rows = []
    for name in names:
        s, preds = d[(d.config == name) & (d.panel == "ensemble")], seed_averaged(name)
        for epoch in E5B["ci_epochs"]:
            cells = []
            for k, t in SET_NAMES.items():
                auroc = s[(s.epoch == epoch) & (s.test_set == t)].auroc.mean()
                i = preds["epochs"].index(epoch)
                delta = fmt_delta(paired_delta(preds[k][0], preds[k][1][i], base[k][1][base["epochs"].index(epoch)]))
                cells.append(f"{auroc:.3f}, Δ {delta}")
            rows.append([name, epoch, *cells])
    return L + table(["Config", "Epoch", *SET_NAMES.values()], rows)


def summary_md(gate, d, lr, path):
    L = [f"# E5b summary\n\n_Generated {datetime.now():%Y-%m-%d %H:%M} by `src/e5b_report.py`._\n",
         "CIs ignore repeated samples per patient (no subject IDs ship with the data), so they are too narrow. The AGP-IBD "
         "hold-out shares patients with the training rows, so in-domain numbers are an upper bound.\n"]
    L += gate_section(gate) + sweep_section(d) + lr_section(lr) + ablation_section(d)
    path.write_text("\n".join(L) + "\n")


def main():
    gate, d, lr = load(GATE_CSV), load(HEAD_CSV), load(LR_CSV)
    own = own_heads(gate)
    if d is not None:
        if "pope" in set(d.config):
            fig_epochs(d, own, FIGURES / "fig_e5b_epochs.png")
        fig_ablation(d, own, FIGURES / "fig_e5b_ablation.png")
    if lr is not None:
        fig_lr(lr, own, FIGURES / "fig_e5b_lr.png")
    summary_md(gate, d, lr, RESULTS / "e5b_summary.md")
    print((RESULTS / "e5b_summary.md").read_text())


if __name__ == "__main__":
    main()
