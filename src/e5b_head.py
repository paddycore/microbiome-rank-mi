"""E5b: why does Pope's head transfer when a fresh linear head on the same features does not? STOP POINTS SB-0..SB-2.

Every head here is trained on frozen features: the cached [CLS] of hidden state 5 (src/e5_states.py), the vector
Pope's head reads. Bodies: ens1..ens10 (Pope's fine-tuned models), pre120 (the pretrained checkpoint they were
fine-tuned from), pre0 (untrained).

Pope's recipe, read from finetune_discriminator/ (begin.py, pretrain_hf.py, dataset.py, run_agp_HF_SH_cross_gen_*.sh):
  head      transformers' ElectraClassificationHead: dropout, dense 200 -> 200, GELU, dropout, dense 200 -> 2
  init      weights N(0, 0.02), biases 0: the pretrained checkpoints have no classifier.* weights, so HF initialises
            them (initializer_range). The shipped heads' dense layer has RMS 0.0205, so one epoch barely moved it.
  dropout   0.2, from the shipped checkpoints' config.json (begin.py's ElectraConfig would default to 0.1)
  training  MSE between softmax[:, 1] and the label, SGD lr 0.01 momentum 0.9, no weight decay, batch 32, weighted
            sampler with replacement, on ALL of AGP-IBD (268 steps per epoch), 1 epoch (the shipped models: epoch0)

What a head trained on cached features cannot reproduce (first suspects if SB-0 fails with every test passing):
  - Pope's head saw noisy features: the frozen encoder ran in train mode (dropout 0.2 inside it) and dataset.py drops
    10% of the input tokens. The cache holds clean features (eval mode, no input dropout).
  - Pope's head moved together with the body's unfrozen parts (position embeddings, LayerNorm, projection, CLS row).
    A fresh head only sees where the body ended up.

Sweeps train on the 80% of AGP-IBD outside the hold-out, so one of their epochs is 215 steps, not Pope's 268: compare
across the two by the `steps` column, not by `epoch`.

Runs:
  python src/e5b_head.py --gate      SB-0: Pope's recipe, 1 epoch, against Pope's own heads    -> results/e5b_gate.csv
  python src/e5b_head.py --configs pope                     SB-1: epoch sweep (HB1)            -> results/e5b.csv
  python src/e5b_head.py --lr                               SB-1: C x scaler sweep (HB2, HB3)  -> results/e5b_lr.csv
  python src/e5b_head.py --configs linear ce class_weight   SB-2: ablations (HB4); no_dropout and default_init also exist
Ensemble-level predictions of each sweep go to results/preds/e5b_<config>_<train_on>.npz. Ensemble = the mean P(IBD)
of ens1..ens10 at one seed, which is how Pope scores.
"""
import argparse
import math
import time
from dataclasses import dataclass, replace
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

import pope_model as pm
from _paths import RESULTS
from e5_states import state_path
from evaluate import bootstrap_ci

E5B = pm.CFG["e5b"]
MODELS = [f"ens{i}" for i in range(1, 11)]
ALL_MODELS = MODELS + E5B["control_models"]
TEST_COHORTS = pm.CFG["e4"]["test_cohorts"]
HOLDOUT = "agp_holdout"
SET_NAMES = {**{c: pm.CFG["cohorts"][c]["name"] for c in TEST_COHORTS}, HOLDOUT: "AGP-IBD hold-out"}
GATE_CSV, HEAD_CSV, LR_CSV = RESULTS / "e5b_gate.csv", RESULTS / "e5b.csv", RESULTS / "e5b_lr.csv"
PRED_DIR = RESULTS / "preds"
HEAD_COLUMNS = ["config", "train_on", "model", "seed", "epoch", "steps", "test_set", "auroc", "auroc_ci_lo",
                "auroc_ci_hi", "n_train", "n_test", "timestamp"]
LR_COLUMNS = ["C", "scale", "model", "test_set", "auroc", "auroc_ci_lo", "auroc_ci_hi", "n_train", "n_test", "timestamp"]


@dataclass(frozen=True)
class HeadConfig:
    kind: str                # "mlp" or "linear"
    loss: str                # "mse" on softmax[:, 1], or "ce" on the logits
    balance: str             # "sampler" or "class_weight"
    init_std: float | None   # None = PyTorch's default nn.Linear init
    dropout: float
    lr: float
    momentum: float
    batch_size: int
    epochs: int


POPE = HeadConfig(**E5B["pope"], epochs=E5B["epochs"])
CONFIGS = {"pope": POPE,                                     # each ablation changes one thing (HB4)
           "linear": replace(POPE, kind="linear"),
           "ce": replace(POPE, loss="ce"),
           "class_weight": replace(POPE, balance="class_weight"),
           "no_dropout": replace(POPE, dropout=0.0),
           "default_init": replace(POPE, init_std=None)}


# --- yours -----------------------------------------------------------------------------------------------------

def make_head(cfg, d):
    """A fresh nn.Module taking (b, d) float features and returning (b, 2) logits.

    cfg.kind "mlp":     Dropout(cfg.dropout), Linear(d, d), GELU(), Dropout(cfg.dropout), Linear(d, 2)
    cfg.kind "linear":  Dropout(cfg.dropout), Linear(d, 2)
    Init: if cfg.init_std is not None, every Linear gets weight ~ Normal(0, cfg.init_std) and bias = 0;
          if it is None, leave PyTorch's default init alone.
    All random draws come from torch's global RNG.
    """
    if cfg.kind == "mlp" :
        layers = [
            nn.Dropout(cfg.dropout),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, 2),
        ]
    elif cfg.kind == "linear": 
        layers = [nn.Dropout(cfg.dropout), nn.Linear(d,2)]
    else :
        raise ValueError(f"Unknown kind: {cfg.kind}")

    head = nn.Sequential(*layers)

    if cfg.init_std is not None:
        for m in head.modules():
          if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=cfg.init_std)
            if m.bias is not None:
              nn.init.zeros_(m.bias)

    return head




def head_loss(logits, y, loss, weights=None):
    """Loss of one batch. logits (b, 2), y (b,) long, weights None or a tensor with one weight per class.

    loss "mse":  e_i = (softmax(logits, 1)[i, 1] - y_i) ** 2, y as float
    loss "ce":   e_i = cross entropy of row i on the raw logits
    weights None:  the mean of e_i
    weights given: sum(w[y_i] * e_i) / sum(w[y_i]), which is what nn.CrossEntropyLoss(weight=w) computes
    """
    if loss == "mse":
        probs = torch.softmax(logits, dim=1)[:, 1]
        e = (probs - y.float()) ** 2
    elif loss == "ce":
        e = F.cross_entropy(logits, y, reduction="none")
    else:
        raise ValueError(f"Unknown loss: {loss}")

    if weights is None:
        return e.mean()
    else:
        w = weights[y]

    return (w * e).sum() / w.sum()


def train_head(X, y, cfg, seed, on_epoch=None):
    """Train a fresh head on frozen features and return it.

    X, y      numpy: (n, d) float32 and (n,) int labels
    seed      first torch.manual_seed(seed), then make_head(cfg, d), so init, dropout and batch order all follow `seed`
    optimizer torch.optim.SGD(lr=cfg.lr, momentum=cfg.momentum), no weight decay
    batches   cfg.batch_size rows, last short batch kept, so ceil(n / batch_size) steps per epoch; a new draw per epoch:
                balance "sampler":       n rows WITH replacement, P(row) proportional to 1 / count(row's class)
                                         (Pope's create_weighted_sampler); head_loss gets weights=None
                balance "class_weight":  every row once, in a new random order; head_loss gets weights = 1 / class counts
    loss      head_loss(logits, y_batch, cfg.loss, weights), called once per step; call the head as head(x)
    on_epoch  if given, on_epoch(epoch, head) at epoch 0 (before any update) and after each of epochs 1..cfg.epochs.
              It may leave the head in eval mode (predict_ibd does), and training must run with dropout on regardless.
    """
    torch.manual_seed(seed)
    n, d = X.shape
    head = make_head(cfg, d)
    X_t = torch.as_tensor(X , dtype = torch.float32)
    y_t = torch.as_tensor(y, dtype = torch.long)

    optimizer = torch.optim.SGD(
        head.parameters(), lr=cfg.lr, momentum=cfg.momentum
    )

    class_counts = torch.bincount(y_t).float()
    if cfg.balance == "sampler" :
        sample_weights = 1.0 / class_counts[y_t]
        loss_weights = None
    elif cfg.balance == "class_weight":
        sample_weights = None
        loss_weights = 1.0 / class_counts
    else :
        raise ValueError(f"Unknown balance : {cfg.balance}")

    if on_epoch is not None : 
        on_epoch(0,head)

    batch_size = cfg.batch_size
    for epoch in range(1, cfg.epochs + 1) :
        head.train()

        if cfg.balance == "sampler" :
            indices = torch.multinomial(
                sample_weights , num_samples=n, replacement=True
            )
        else :
            indices = torch.randperm(n)

        for start in range(0 , n, batch_size) :
            batch_idx = indices[start : start + batch_size]
            xb = X_t[batch_idx]
            yb = y_t[batch_idx]

            optimizer.zero_grad()
            logits = head(xb)
            loss = head_loss(logits, yb, cfg.loss, weights=loss_weights)
            loss.backward()
            optimizer.step()

        if on_epoch is not None :
            on_epoch(epoch, head)
        
    return head


def fit_lr(X, y, C, scale):
    """Fit and return make_pipeline(LogisticRegression(C=C, class_weight="balanced", max_iter=5000)), with a
    StandardScaler() step in front when `scale` is true (E4b's fit_head had the scaler always, and sklearn's default C)."""
    clf_lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    if scale:
        clf = make_pipeline(StandardScaler(), clf_lr)
    else:
        clf = make_pipeline(clf_lr)
    clf.fit(X, y)
    return clf


# --- harness ---------------------------------------------------------------------------------------------------

def predict_ibd(head, X):
    """P(IBD) for each row of numpy X, dropout off."""
    head.eval()
    with torch.no_grad():
        return torch.softmax(head(torch.as_tensor(X, dtype=torch.float32)), 1)[:, 1].numpy()


#@lru_cache(maxsize=None)
_BODY_CACHE = {}
def body_features(model):
    """{cohort: (X, y)}: float32 [CLS] of the last hidden state (n, 200) and the labels, in cache order."""
    out = {}
    for c in pm.CFG["e5"]["cohorts"]:
        z = np.load(state_path(model, c))
        out[c] = z["cls"][:, E5B["layer"]].astype(np.float32), z["y"].astype(int)
    return out


def agp_split(y):
    """(train rows, hold-out rows) of AGP-IBD: stratified, the same for every model, seed and config."""
    split = StratifiedShuffleSplit(n_splits=1, test_size=E5B["holdout_frac"], random_state=E5B["split_seed"])
    return next(split.split(np.zeros((len(y), 1)), y))


def eval_sets(feats, train_on):
    """(rows of AGP-IBD to train on, {eval set: (X, y)}). train_on "all" = every row, as Pope; "80" = outside the hold-out."""
    X, y = feats["agp_ibd"]
    sets = {c: feats[c] for c in TEST_COHORTS}
    if train_on == "all":
        return np.arange(len(y)), sets
    rows, hold = agp_split(y)
    return rows, {**sets, HOLDOUT: (X[hold], y[hold])}


def run_seed(model, seed):
    """The seed a head on `model` really gets. It must differ across models: an ensemble whose 10 heads shared one init
    and one batch order would be less diverse than Pope's, whose heads have unrelated inits (pairwise cosine 0.00)."""
    return 1000 * ALL_MODELS.index(model) + seed


def head_cell(model, cfg, seed, train_on):
    """One head on one body. Returns {epoch: {eval set: P(IBD)}} at the configured checkpoints up to cfg.epochs."""
    torch.set_num_threads(1)
    feats = body_features(model)
    rows, sets = eval_sets(feats, train_on)
    X, y = feats["agp_ibd"]
    scores = {}

    def record(epoch, head):
        if epoch in E5B["checkpoints"]:
            scores[epoch] = {k: predict_ibd(head, A) for k, (A, _) in sets.items()}

    train_head(X[rows], y[rows], cfg, run_seed(model, seed), on_epoch=record)
    return scores


def lr_cell(model, C, scale):
    """One logistic regression on one body, trained on the 80%. Returns {eval set: P(IBD)}."""
    feats = body_features(model)
    rows, sets = eval_sets(feats, "80")
    X, y = feats["agp_ibd"]
    clf = fit_lr(X[rows], y[rows], C, scale)
    return {k: clf.predict_proba(A)[:, 1] for k, (A, _) in sets.items()}


def score(y, p, ci=False):
    lo, hi = bootstrap_ci(y, p, seed=0)[0] if ci else ("", "")
    return {"auroc": round(roc_auc_score(y, p), 4), "auroc_ci_lo": round(lo, 4) if ci else "",
            "auroc_ci_hi": round(hi, 4) if ci else ""}


def ensemble_scores(runs):
    """{seed: {epoch: {eval set: mean P(IBD) over ens1..ens10}}}; empty unless every member ran for every seed."""
    seeds = sorted({s for _, s in runs})
    if not all((m, s) in runs for m in MODELS for s in seeds):
        return {}
    return {s: {e: {k: np.mean([runs[m, s][e][k] for m in MODELS], axis=0) for k in sets}
                for e, sets in runs[MODELS[0], s].items()} for s in seeds}


def head_rows(name, train_on, runs, ens, cfg, n_train, labels):
    """CSV rows: one per model x seed x epoch x eval set, plus model "ensemble" (with a CI at the configured epochs)."""
    stamp = datetime.now().isoformat(timespec="seconds")
    per_epoch = math.ceil(n_train / cfg.batch_size)

    def row(model, seed, epoch, key, p, ci=False):
        return {"config": name, "train_on": train_on, "model": model, "seed": seed, "epoch": epoch,
                "steps": epoch * per_epoch, "test_set": SET_NAMES[key], **score(labels[key], p, ci),
                "n_train": n_train, "n_test": len(p), "timestamp": stamp}

    rows = [row(m, s, e, k, p) for (m, s), scores in runs.items() for e, sets in scores.items() for k, p in sets.items()]
    rows += [row("ensemble", s, e, k, p, ci=e in E5B["ci_epochs"])
             for s, scores in ens.items() for e, sets in scores.items() for k, p in sets.items()]
    return rows


def labels_of(train_on):
    """(number of training rows, {eval set: y})."""
    rows, sets = eval_sets(body_features(MODELS[0]), train_on)
    return len(rows), {k: y for k, (_, y) in sets.items()}


def run_config(name, cfg, models, train_on, jobs):
    """Every model x seed for one config. Returns (CSV rows, ensemble scores, labels)."""
    grid = [(m, s) for m in models for s in E5B["seeds"]]
    runs = dict(zip(grid, Parallel(n_jobs=jobs, verbose=5)(delayed(head_cell)(m, cfg, s, train_on) for m, s in grid)))
    ens = ensemble_scores(runs)
    n_train, labels = labels_of(train_on)
    return head_rows(name, train_on, runs, ens, cfg, n_train, labels), ens, labels


def merge_csv(path, rows, columns, key):
    """Write `rows`, keeping earlier rows unless they share `key` values with a new row (a re-run replaces its config)."""
    new = pd.DataFrame(rows, columns=columns).astype(str)
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        seen = set(map(tuple, new[key].to_numpy()))
        new = pd.concat([old[[tuple(r) not in seen for r in old[key].to_numpy()]], new])
    new.to_csv(path, index=False)


def save_preds(name, train_on, ens, labels):
    """results/preds/e5b_<config>_<train_on>.npz: seeds, epochs, <set>_y (n,), <set>_p (seeds, epochs, n), ensemble level."""
    seeds, epochs = sorted(ens), sorted(next(iter(ens.values())))
    arrays = {"seeds": seeds, "epochs": epochs}
    for k, y in labels.items():
        arrays[f"{k}_y"] = y
        arrays[f"{k}_p"] = np.array([[ens[s][e][k] for e in epochs] for s in seeds], dtype=np.float32)
    np.savez_compressed(PRED_DIR / f"e5b_{name}_{train_on}.npz", **arrays)


# --- SB-0 ------------------------------------------------------------------------------------------------------

def pope_head_state(model):
    """The classifier.* weights of the shipped run behind ens<i>."""
    sd = torch.load(pm.ensemble_dirs()[int(model[3:]) - 1] / "pytorch_model.bin", map_location="cpu", weights_only=True)
    return {k: sd[f"classifier.{k}"] for k in ("dense.weight", "dense.bias", "out_proj.weight", "out_proj.bias")}


def pope_head_probs(state, X):
    """P(IBD) from one of Pope's own heads on numpy features X (no dropout, as at test time)."""
    x = torch.as_tensor(X, dtype=torch.float32)
    h = F.gelu(F.linear(x, state["dense.weight"], state["dense.bias"]))
    return torch.softmax(F.linear(h, state["out_proj.weight"], state["out_proj.bias"]), 1)[:, 1].numpy()


def gate_passes(fresh_ensemble_aurocs, own_auroc):
    """SB-0, one cohort: the mean over seeds of the fresh ensemble's AUROC is within tolerance of Pope's own heads."""
    return abs(np.mean(fresh_ensemble_aurocs) - own_auroc) <= pm.CFG["e4"]["gate_tolerance"]


def own_head_rows():
    """Pope's own heads on the cached (float16) features, with the difference from E4's stored P(IBD) printed."""
    stamp = datetime.now().isoformat(timespec="seconds")
    states = {m: pope_head_state(m) for m in MODELS}
    n_train, labels = labels_of("all")
    rows = []
    for c in TEST_COHORTS:
        probs = np.stack([pope_head_probs(states[m], body_features(m)[c][0]) for m in MODELS])
        stored = np.load(PRED_DIR / f"e4_sorted_r0_{c}.npz")["probs"]
        print(f"{SET_NAMES[c]}: Pope's heads on the cache vs E4's stored P(IBD), largest difference "
              f"{np.abs(probs - stored).max():.1e}", flush=True)
        for model, p, ci in [*zip(MODELS, probs, [False] * len(MODELS)), ("ensemble", probs.mean(0), True)]:
            rows.append({"config": "pope_own_head", "train_on": "all", "model": model, "seed": "", "epoch": 1,
                         "steps": math.ceil(n_train / POPE.batch_size), "test_set": SET_NAMES[c],
                         **score(labels[c], p, ci), "n_train": n_train, "n_test": len(p), "timestamp": stamp})
    return rows


def print_gate(rows):
    d = pd.DataFrame(rows)
    pope = pm.CFG["e4"]["pope_auroc"]
    print(f"\nSB-0: |fresh ensemble - Pope's own heads| <= {pm.CFG['e4']['gate_tolerance']}, per cohort, fresh heads on all of "
          "AGP-IBD, 1 epoch. epoch 0 = untrained heads. ensemble = mean [min, max] over seeds, member = mean ± sd.")
    print(f"{'set':17s} {'heads':24s} {'ensemble':>23s} {'member':>15s}  verdict")
    for key, name in SET_NAMES.items():
        own = d[(d.config == "pope_own_head") & (d.test_set == name)]
        if len(own):
            ens, mem = own[own.model == "ensemble"].auroc.iloc[0], own[own.model != "ensemble"].auroc
            print(f"{name:17s} {f'Pope (paper {pope[key]:.3f})':24s} {ens:23.3f} {f'{mem.mean():.3f} ± {mem.std():.3f}':>15s}")
        for train_on, epoch in (("all", 0), ("all", 1), ("80", 0), ("80", 1)):
            r = d[(d.config == "pope") & (d.train_on == train_on) & (d.epoch == epoch) & (d.test_set == name)]
            if r.empty:
                continue
            e, m = r[r.model == "ensemble"].auroc, r[r.model != "ensemble"].auroc
            verdict = ""
            if len(own) and (train_on, epoch) == ("all", 1):
                verdict = "PASS" if gate_passes(e, ens) else "FAIL"
            print(f"{name:17s} {f'fresh, {train_on}, epoch {epoch}':24s} "
                  f"{f'{e.mean():.3f} [{e.min():.3f}, {e.max():.3f}]':>23s} {f'{m.mean():.3f} ± {m.std():.3f}':>15s}  {verdict}")


def gate(jobs):
    cfg = replace(POPE, epochs=1)
    rows = own_head_rows()
    for train_on in ("all", "80"):
        rows += run_config("pope", cfg, MODELS, train_on, jobs)[0]
    pd.DataFrame(rows, columns=HEAD_COLUMNS).to_csv(GATE_CSV, index=False)
    print_gate(rows)
    print(f"{len(rows)} rows -> {GATE_CSV}")


# --- SB-1, SB-2 ------------------------------------------------------------------------------------------------

def lr_sweep(models, jobs):
    grid = [(m, C, scale) for m in models for C in E5B["C"] for scale in (True, False)]
    runs = dict(zip(grid, Parallel(n_jobs=jobs, verbose=5)(delayed(lr_cell)(*g) for g in grid)))
    n_train, labels = labels_of("80")
    stamp = datetime.now().isoformat(timespec="seconds")

    def row(model, C, scale, key, p, ci=False):
        return {"C": C, "scale": scale, "model": model, "test_set": SET_NAMES[key], **score(labels[key], p, ci),
                "n_train": n_train, "n_test": len(p), "timestamp": stamp}

    rows = [row(m, C, scale, k, p) for (m, C, scale), sets in runs.items() for k, p in sets.items()]
    if set(MODELS) <= set(models):
        rows += [row("ensemble", C, scale, k, np.mean([runs[m, C, scale][k] for m in MODELS], axis=0), ci=True)
                 for C in E5B["C"] for scale in (True, False) for k in labels]
    pd.DataFrame(rows, columns=LR_COLUMNS).to_csv(LR_CSV, index=False)
    print(f"{len(rows)} rows -> {LR_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="SB-0: Pope's recipe for 1 epoch, against Pope's own heads")
    ap.add_argument("--configs", nargs="+", default=[], choices=list(CONFIGS), help="epoch sweep of these configs")
    ap.add_argument("--lr", action="store_true", help="logistic regression sweep over C, with and without a scaler")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    if args.gate:
        gate(args.jobs)
    for name in args.configs:
        rows, ens, labels = run_config(name, CONFIGS[name], args.models, "80", args.jobs)
        merge_csv(HEAD_CSV, rows, HEAD_COLUMNS, ["config", "train_on"])
        if ens:
            save_preds(name, "80", ens, labels)
        print(f"{name}: {len(rows)} rows -> {HEAD_CSV}  ({time.time() - t0:.0f}s so far)", flush=True)
    if args.lr:
        lr_sweep(args.models, args.jobs)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
