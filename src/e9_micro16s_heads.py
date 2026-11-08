"""E9 step 2: E5b's head experiments on Micro16S features (notebook/plan_e9.md).

Input as in the Micro16S paper: Pope's full coordinate list per sample (up to 1,897 entries), the 768 most abundant
entries, relative abundance renormalised within them, each ASV replaced by its Micro16S embedding
(src/e9_micro16s_embed.py). Three frozen bodies, 256-d each:

  pre    the pretrained Micro16S transformer (checkpoint_epoch_016), mean of its token outputs
  rand   the same architecture, randomly initialised (seed 0): the counterpart of pre0
  bag    the abundance-weighted mean of the ASV embeddings, no transformer

Heads: Pope's recipe (e5b_head.train_head, POPE) on features standardised with the AGP 80% statistics (the three bodies
have different scales), 10 seeds, E5b's checkpoints; "seed ensemble" = mean P(IBD) over the 10 seeds. LR sweep: E5b's
grid of C with and without a scaler.

Usage: python src/e9_micro16s_heads.py --extract   (~10-20 min on CPU; existing files are skipped)
       python src/e9_micro16s_heads.py --heads --lr --report
"""
import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from matplotlib.lines import Line2D

import e5b_head as h
import e5b_report as rep
import pope_model as pm
import report as p0
from _paths import FIGURES, RESULTS
from data import COHORTS as RAW_FILES
from e9_micro16s_embed import MICRO16S_DIR, OUT as ASV_EMBEDDINGS

BODIES = ("pre", "rand", "bag")
SEEDS = range(10)
K = 768
TRANSFORMER_CKPT = MICRO16S_DIR / "micro16s-transformer/m16s_transformer_runs/pretrain_001/checkpoints/checkpoint_epoch_016.pth"
HEAD_CSV, LR_CSV = RESULTS / "e9.csv", RESULTS / "e9_lr.csv"
LABELS = {"pre": "A  Micro16S, pretrained transformer", "rand": "B  Micro16S, random-init transformer",
          "bag": "C  Micro16S embeddings, no transformer", "pope": "D  Pope's bodies (E5b, for comparison)"}


def path(body, cohort):
    return pm.path("features_dir") / f"e9_{body}_{cohort}.npz"


def profile_batches(cohort, emb, bs=64):
    """(E (b, K, 256), A (b, K), pad mask (b, K)) per batch: top-K entries by count, abundance renormalised."""
    arr = np.load(pm.path("cohorts_dir") / RAW_FILES[cohort][0], mmap_mode="r")
    for s in range(0, len(arr), bs):
        chunk = np.asarray(arr[s:s + bs])
        E = np.zeros((len(chunk), K, emb.shape[1]), np.float32)
        A = np.zeros((len(chunk), K), np.float32)
        M = np.ones((len(chunk), K), bool)
        for i, row in enumerate(chunk):
            row = row[row[:, 1] > 0]
            row = row[np.argsort(-row[:, 1], kind="stable")][:K]
            n = len(row)
            E[i, :n], A[i, :n], M[i, :n] = emb[row[:, 0].astype(int)], row[:, 1] / row[:, 1].sum(), False
        yield E, A, M


def transformer(body):
    sys.path.append(str(MICRO16S_DIR / "micro16s-transformer"))       # appended: never shadows this project's modules
    from Micro16STransformer import Micro16STransformer
    ckpt = torch.load(TRANSFORMER_CKPT, map_location="cpu", weights_only=True)
    torch.manual_seed(0)
    model = Micro16STransformer(**ckpt["config"])
    if body == "pre":
        model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


@torch.no_grad()
def extract():
    emb = np.load(ASV_EMBEDDINGS)
    meta = pd.read_csv(RESULTS / "e5_meta.csv")
    for body in BODIES:
        model = None if body == "bag" else transformer(body)
        for c in pm.CFG["e5"]["cohorts"]:
            if path(body, c).exists():
                continue
            t0, parts = time.time(), []
            for E, A, M in profile_batches(c, emb):
                if model is None:
                    parts.append((A[..., None] * E).sum(1))
                else:
                    parts.append(model.get_embedding_representations(torch.from_numpy(E), torch.from_numpy(A),
                                                                     torch.from_numpy(M), aggregate="mean").numpy())
            X = np.concatenate(parts).astype(np.float32)
            np.savez_compressed(path(body, c), X=X, y=meta[meta.cohort == c].y.to_numpy())
            print(f"{body:4s} {c:8s} {X.shape} {time.time() - t0:.0f}s", flush=True)


def features(body, standardise=True):
    """{cohort: (X, y)}, standardised with the mean and sd of the AGP 80% training rows."""
    out = {c: (np.load(path(body, c))["X"], np.load(path(body, c))["y"]) for c in pm.CFG["e5"]["cohorts"]}
    if standardise:
        rows, _ = h.eval_sets(out, "80")
        X = out["agp_ibd"][0][rows]
        mu, sd = X.mean(0), X.std(0) + 1e-8
        out = {c: ((X_ - mu) / sd, y) for c, (X_, y) in out.items()}
    return out


def cell(body, seed):
    torch.set_num_threads(1)
    feats = features(body)
    rows, sets = h.eval_sets(feats, "80")
    X, y = feats["agp_ibd"]
    scores = {}

    def record(epoch, head):
        if epoch in h.E5B["checkpoints"]:
            scores[epoch] = {k: h.predict_ibd(head, A) for k, (A, _) in sets.items()}

    h.train_head(X[rows], y[rows], h.POPE, 1000 * BODIES.index(body) + seed, on_epoch=record)
    return scores


def heads(jobs):
    grid = [(b, s) for b in BODIES for s in SEEDS]
    runs = dict(zip(grid, Parallel(n_jobs=jobs)(delayed(cell)(b, s) for b, s in grid)))
    _, labels = h.labels_of("80")
    rows = []
    for body in BODIES:
        for epoch in h.E5B["checkpoints"]:
            for k, y in labels.items():
                ps = [runs[body, s][epoch][k] for s in SEEDS]
                rows += [dict(body=body, seed=s, epoch=epoch, test_set=h.SET_NAMES[k], **h.score(y, p)) for s, p in zip(SEEDS, ps)]
                rows.append(dict(body=body, seed="ensemble", epoch=epoch, test_set=h.SET_NAMES[k],
                                 **h.score(y, np.mean(ps, 0), ci=epoch in h.E5B["ci_epochs"])))
    pd.DataFrame(rows).to_csv(HEAD_CSV, index=False)
    print(f"{len(rows)} rows -> {HEAD_CSV}")


def lr(jobs):
    def one(body, C, scale):
        feats = features(body, standardise=False)
        rows, sets = h.eval_sets(feats, "80")
        X, y = feats["agp_ibd"]
        clf = h.fit_lr(X[rows], y[rows], C, scale)
        return [dict(body=body, C=C, scale=scale, test_set=h.SET_NAMES[k], **h.score(yk, clf.predict_proba(A)[:, 1]))
                for k, (A, yk) in sets.items()]

    out = Parallel(n_jobs=jobs)(delayed(one)(b, C, s) for b in BODIES for C in h.E5B["C"] for s in (True, False))
    pd.DataFrame([r for rows in out for r in rows]).to_csv(LR_CSV, index=False)
    print(f"-> {LR_CSV}")


def report():
    d = pd.read_csv(HEAD_CSV)
    d["auroc"] = pd.to_numeric(d.auroc)
    pope = pd.read_csv(h.HEAD_CSV, keep_default_na=False)
    pope = pope[(pope.config == "pope") & pope.model.str.startswith("ens") & (pope.model != "ensemble")].copy()
    pope["auroc"] = pd.to_numeric(pope.auroc)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, facecolor=p0.SURFACE)
    for ax, (body, title) in zip(axes.ravel(), LABELS.items()):
        p0.style(ax)
        ax.axvline(1, color=p0.GRID, lw=1, zorder=1)
        s = pope if body == "pope" else d[(d.body == body) & (d.seed != "ensemble")]
        ends = {}
        for test_set, color in rep.COLORS.items():
            r = rep.curve(s[s.test_set == test_set], "epoch")
            rep.draw(ax, r, "epoch", color)
            ends[test_set] = r["mean"].iloc[-1]
        rep.finish(ax, title, "single heads: mean, band = min-max over " + ("10 bodies x 5 seeds" if body == "pope" else "10 seeds"))
        rep.end_labels(ax, ends, 50)
        rep.epoch_axis(ax)
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC", color=p0.INK, fontsize=9)
    fig.legend(handles=[Line2D([], [], color=c, lw=2, marker="o", ms=6, label=k) for k, c in rep.COLORS.items()],
               loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=p0.INK)
    rep.save(fig, FIGURES / "fig_e9_micro16s_epochs.png",
             "Heads with Pope's recipe (E5b) on frozen, standardised features, trained on 6,856 AGP-IBD rows. Micro16S input: top-768 "
             "ASVs of Pope's full lists, Micro16S embeddings + relative abundance. Vertical line = 1 epoch. Dashed = chance.")
    t = d[d.seed != "ensemble"].groupby(["body", "test_set", "epoch"]).auroc.mean().unstack("epoch")[[0, 1, 2, 5, 10, 20, 50]].round(3)
    e = d[d.seed == "ensemble"].groupby(["body", "test_set", "epoch"]).auroc.mean().unstack("epoch")[[0, 1, 2, 5, 10, 20, 50]].round(3)
    l = pd.read_csv(LR_CSV).pivot_table(index=["body", "scale", "C"], columns="test_set", values="auroc").round(3)
    block = lambda df: "```\n" + df.to_string() + "\n```"
    text = ["# E9 summary: E5b on Micro16S\n", "## Single heads, mean over 10 seeds\n", block(t),
            "\n## Seed ensemble (mean P(IBD) over 10 seeds)\n", block(e), "\n## Logistic regression sweep\n", block(l)]
    (RESULTS / "e9_summary.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for flag in ("extract", "heads", "lr", "report"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    if args.extract:
        extract()
    if args.heads:
        heads(args.jobs)
    if args.lr:
        lr(args.jobs)
    if args.report:
        report()
