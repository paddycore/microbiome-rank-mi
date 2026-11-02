"""Tests for src/e5b_head.py (run: pytest src/test_e5b_head.py).

make_head, head_loss, train_head and fit_lr are yours to write, and these tests pin down how they behave. Everything
below "bookkeeping" checks the harness on fake or value-coded features, so it needs no real cache.
"""
import copy
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from transformers import ElectraConfig
from transformers.models.electra.modeling_electra import ElectraClassificationHead

import e5b_head
from e5b_head import E5B, MODELS, POPE, fit_lr, head_loss, make_head, predict_ibd, train_head


def linears(head):
    return [m for m in head.modules() if isinstance(m, nn.Linear)]


def weights_of(head):
    return {k: v.detach().clone() for k, v in head.state_dict().items()}


def same_weights(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def features(n=400, pos=0.05, d=6, sep=3.0, seed=0):
    """Column 0 carries the label, column 1 = row number / n (so a test can tell which rows were fed)."""
    rng = np.random.default_rng(seed)
    y = rng.permutation((np.arange(n) < round(n * pos)).astype(int))
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[:, 0] += sep * y
    X[:, 1] = np.arange(n) / n
    return X, y


class Spy:
    """on_epoch callback: hooks the head at epoch 0, then records every batch fed to it, per epoch."""

    def __init__(self, set_eval=False):
        self.set_eval, self.calls, self.batches, self.training, self.epoch = set_eval, [], {}, [], 0

    def __call__(self, epoch, head):
        self.calls.append(epoch)
        if epoch == 0:
            head.register_forward_pre_hook(self.hook)
        self.epoch = epoch + 1
        if self.set_eval:
            head.eval()

    def hook(self, module, inputs):
        self.batches.setdefault(self.epoch, []).append(inputs[0].detach().clone())
        self.training.append(module.training)

    def rows(self, epoch, n):
        return np.round(torch.cat(self.batches[epoch])[:, 1].numpy() * n).astype(int)


# --- make_head -------------------------------------------------------------------------------------------------

def test_make_head_mlp_is_electra_classification_head():
    torch.manual_seed(0)
    head = make_head(POPE, 200).eval()
    ref = ElectraClassificationHead(ElectraConfig(hidden_size=200, num_labels=2)).eval()
    dense, out = linears(head)
    ref.dense.load_state_dict(dense.state_dict())
    ref.out_proj.load_state_dict(out.state_dict())
    x = torch.randn(16, 200)
    with torch.no_grad():
        torch.testing.assert_close(head(x), ref(x[:, None]))


def test_make_head_linear_is_one_layer():
    head = make_head(replace(POPE, kind="linear"), 7)
    assert [(m.in_features, m.out_features) for m in linears(head)] == [(7, 2)]
    assert head(torch.zeros(3, 7)).shape == (3, 2)


@pytest.mark.parametrize("kind, n", [("mlp", 2), ("linear", 1)])
def test_make_head_dropout_layers(kind, n):
    head = make_head(replace(POPE, kind=kind, dropout=0.35), 8)
    assert [m.p for m in head.modules() if isinstance(m, nn.Dropout)] == [0.35] * n


def test_make_head_dropout_comes_first():
    # p = 1 zeroes the input, so the logits cannot depend on it (a head without the leading dropout would still see x)
    head = make_head(replace(POPE, kind="linear", dropout=1.0, init_std=0.5), 8).train()
    out = head(torch.randn(4, 8))
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_make_head_init():
    torch.manual_seed(0)
    dense, out = linears(make_head(POPE, 200))
    assert dense.weight.std().item() == pytest.approx(0.02, rel=0.03)
    assert out.weight.std().item() == pytest.approx(0.02, rel=0.25)
    assert (dense.bias == 0).all() and (out.bias == 0).all()


def test_make_head_none_keeps_pytorch_default_init():
    torch.manual_seed(0)
    dense, out = linears(make_head(replace(POPE, init_std=None), 200))
    assert dense.weight.std().item() > 0.03          # default: uniform(-1/sqrt(d), 1/sqrt(d)), std about 0.041
    assert (dense.bias != 0).any() and (out.bias != 0).any()


def test_make_head_draws_from_global_rng():
    torch.manual_seed(1)
    a = weights_of(make_head(POPE, 20))
    b = weights_of(make_head(POPE, 20))
    torch.manual_seed(1)
    assert same_weights(a, weights_of(make_head(POPE, 20)))
    assert not same_weights(a, b)


# --- head_loss -------------------------------------------------------------------------------------------------

LOGITS = torch.tensor([[0.0, 1.0], [2.0, -1.0], [0.5, 0.5], [-1.0, 3.0]])
LABELS = torch.tensor([1, 0, 1, 0])
P1 = np.exp(LOGITS.numpy()[:, 1]) / np.exp(LOGITS.numpy()).sum(1)
SQ = (P1 - LABELS.numpy()) ** 2
CE = -np.log(np.where(LABELS.numpy() == 1, P1, 1 - P1))
W = torch.tensor([1.0, 4.0])
W_ROW = W.numpy()[LABELS.numpy()]


@pytest.mark.parametrize("loss, weights, expected", [
    ("mse", None, SQ.mean()),                                # squared error of softmax[:, 1], not of the raw logits
    ("ce", None, CE.mean()),
    ("mse", W, (W_ROW * SQ).sum() / W_ROW.sum()),            # weighted mean: divided by the weights, not by b
    ("ce", W, (W_ROW * CE).sum() / W_ROW.sum()),
])
def test_head_loss_values(loss, weights, expected):
    assert head_loss(LOGITS, LABELS, loss, weights).item() == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("loss", ["mse", "ce"])
def test_head_loss_is_differentiable_and_scale_free_in_weights(loss):
    logits = LOGITS.clone().requires_grad_()
    value = head_loss(logits, LABELS, loss, W)
    value.backward()
    assert logits.grad.abs().sum() > 0
    assert head_loss(LOGITS, LABELS, loss, 10 * W).item() == pytest.approx(value.item(), rel=1e-5)


# --- train_head ------------------------------------------------------------------------------------------------

SMALL = replace(POPE, epochs=2)


@pytest.mark.parametrize("balance", ["sampler", "class_weight"])
def test_train_head_calls_on_epoch_from_0(balance):
    X, y = features(n=64)
    spy = Spy()
    train_head(X, y, replace(POPE, balance=balance, epochs=3), 0, on_epoch=spy)
    assert spy.calls == [0, 1, 2, 3]


def test_train_head_epoch_0_is_the_untouched_init():
    X, y = features(n=64)
    snaps = {}
    train_head(X, y, SMALL, 5, on_epoch=lambda e, head: snaps.setdefault(e, weights_of(head)))
    torch.manual_seed(5)
    assert same_weights(snaps[0], weights_of(make_head(SMALL, X.shape[1])))
    assert not same_weights(snaps[0], snaps[1]) and not same_weights(snaps[1], snaps[2])


def test_train_head_returns_the_trained_head_and_on_epoch_is_optional():
    X, y = features(n=64)
    snaps = {}
    head = train_head(X, y, SMALL, 3, on_epoch=lambda e, h: snaps.setdefault(e, weights_of(h)))
    assert same_weights(weights_of(head), snaps[2])
    assert same_weights(weights_of(train_head(X, y, SMALL, 3)), snaps[2])


@pytest.mark.parametrize("balance", ["sampler", "class_weight"])
def test_train_head_steps_and_batch_size(balance):
    # the last short batch is kept: 100 rows at batch 32 = 4 steps per epoch, the last one 4 rows
    X, y = features(n=100, pos=0.2)
    spy = Spy()
    train_head(X, y, replace(SMALL, balance=balance), 0, on_epoch=spy)
    for epoch in (1, 2):
        assert [len(b) for b in spy.batches[epoch]] == [32, 32, 32, 4]


def test_train_head_sampler_balances_the_classes_with_replacement():
    n = 400
    X, y = features(n=n, pos=0.05)
    spy = Spy()
    train_head(X, y, replace(POPE, epochs=30), 0, on_epoch=spy)
    drawn = np.concatenate([spy.rows(e, n) for e in spy.batches])
    assert 0.42 < y[drawn].mean() < 0.58                     # 5% of the rows, half of the draws
    assert len(drawn) == 30 * n
    assert len(set(spy.rows(1, n))) < n                      # with replacement: an epoch does not cover every row
    assert not np.array_equal(spy.rows(1, n), spy.rows(2, n))


def test_train_head_class_weight_visits_every_row_once_per_epoch():
    n = 100
    X, y = features(n=n, pos=0.2)
    spy = Spy()
    train_head(X, y, replace(SMALL, balance="class_weight"), 0, on_epoch=spy)
    for epoch in (1, 2):
        assert sorted(spy.rows(epoch, n)) == list(range(n))
    assert not np.array_equal(spy.rows(1, n), spy.rows(2, n))  # reshuffled each epoch


@pytest.mark.parametrize("balance", ["sampler", "class_weight"])
def test_train_head_keeps_dropout_on_after_an_eval_callback(balance):
    # predict_ibd in on_epoch leaves the head in eval mode: dropout must still be on in every later step
    X, y = features(n=64)
    spy = Spy(set_eval=True)
    train_head(X, y, replace(POPE, balance=balance, epochs=3), 0, on_epoch=spy)
    assert len(spy.training) == 3 * 2 and all(spy.training)


@pytest.mark.parametrize("balance", ["sampler", "class_weight"])
def test_train_head_is_reproducible_from_its_seed(balance):
    X, y = features(n=64)
    cfg = replace(SMALL, balance=balance)
    a = weights_of(train_head(X, y, cfg, 7))
    torch.manual_seed(999)                                   # whatever the RNG state was before the call
    assert same_weights(a, weights_of(train_head(X, y, cfg, 7)))
    assert not same_weights(a, weights_of(train_head(X, y, cfg, 8)))


@pytest.mark.parametrize("balance", ["sampler", "class_weight"])
def test_train_head_batch_order_follows_the_seed(balance):
    n = 100
    X, y = features(n=n, pos=0.2)
    orders = []
    for seed in (1, 1, 2):
        spy = Spy()
        train_head(X, y, replace(SMALL, balance=balance), seed, on_epoch=spy)
        orders.append(spy.rows(1, n))
    assert np.array_equal(orders[0], orders[1]) and not np.array_equal(orders[0], orders[2])


@pytest.mark.parametrize("balance", ["sampler", "class_weight"])
@pytest.mark.parametrize("loss", ["mse", "ce"])
def test_train_head_matches_the_updates_done_by_hand(loss, balance):
    # with no dropout, training depends only on the batches, so replaying the batches it fed must give the same weights:
    # velocity = momentum * velocity + grad, weight -= lr * velocity, once per batch. Catches zero_grad or step outside
    # the batch loop, Adam, weight decay, lost momentum, and the wrong loss or class weights.
    n = 60
    X, y = features(n=n, pos=0.2)
    cfg = replace(POPE, loss=loss, balance=balance, dropout=0.0, batch_size=16, epochs=2, lr=0.05, momentum=0.8)
    start, spy = {}, Spy()

    def on_epoch(epoch, head):
        start.setdefault("head", copy.deepcopy(head))        # taken before spy hooks the head
        spy(epoch, head)

    final = train_head(X, y, cfg, 0, on_epoch=on_epoch)
    head, yt = start["head"].train(), torch.as_tensor(y).long()
    params = list(head.parameters())
    velocity = [torch.zeros_like(p) for p in params]
    class_w = 1 / torch.bincount(yt).float()
    for epoch in (1, 2):
        assert len(spy.batches[epoch]) == 4
        for xb in spy.batches[epoch]:
            yb = yt[torch.round(xb[:, 1] * n).long()]
            logits = head(xb)
            if loss == "mse":
                e = (torch.softmax(logits, 1)[:, 1] - yb.float()) ** 2
            else:
                e = F.cross_entropy(logits, yb, reduction="none")
            value = e.mean() if balance == "sampler" else (class_w[yb] * e).sum() / class_w[yb].sum()
            grads = torch.autograd.grad(value, params)
            with torch.no_grad():
                for p, v, g in zip(params, velocity, grads):
                    v.mul_(cfg.momentum).add_(g)
                    p.sub_(cfg.lr * v)
    for k, v in head.state_dict().items():
        torch.testing.assert_close(v, final.state_dict()[k], rtol=1e-4, atol=1e-6, msg=lambda m: f"{k}: {m}")


@pytest.mark.parametrize("balance", ["sampler", "class_weight"])
@pytest.mark.parametrize("loss", ["mse", "ce"])
def test_train_head_passes_the_loss_and_weights_on(monkeypatch, balance, loss):
    X, y = features(n=100, pos=0.2)
    seen, real = [], e5b_head.head_loss

    def spy(logits, y_batch, loss_name, weights=None):
        seen.append((loss_name, weights))
        return real(logits, y_batch, loss_name, weights)

    monkeypatch.setattr(e5b_head, "head_loss", spy)
    train_head(X, y, replace(SMALL, balance=balance, loss=loss), 0)
    assert len(seen) == 2 * 4 and {s[0] for s in seen} == {loss}
    for _, weights in seen:
        if balance == "sampler":
            assert weights is None                           # the sampler already balances; weighting too would be twice
        else:
            assert weights[1].item() / weights[0].item() == pytest.approx(80 / 20)


@pytest.mark.parametrize("balance", ["sampler", "class_weight"])
def test_train_head_learns(balance):
    X, y = features(n=400, pos=0.1)
    X_new, y_new = features(n=400, pos=0.1, seed=1)
    head = train_head(X, y, replace(POPE, balance=balance, epochs=40), 0)
    assert roc_auc_score(y_new, predict_ibd(head, X_new)) > 0.9


# --- fit_lr ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("scale, steps", [(True, 2), (False, 1)])
def test_fit_lr_settings(scale, steps):
    X, y = features()
    clf = fit_lr(X, y, 0.3, scale)
    assert len(clf) == steps
    # every other setting must be sklearn's default (positional arguments land in the wrong parameter)
    assert clf[-1].get_params() == LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000).get_params()
    assert isinstance(clf[0], StandardScaler) == scale


def test_fit_lr_scaler_fits_the_training_features():
    X, y = features()
    np.testing.assert_allclose(fit_lr(X, y, 1.0, True)[0].mean_, X.mean(0), rtol=1e-5, atol=1e-5)


def test_fit_lr_scale_decides_whether_units_matter():
    X, y = features()
    probs = lambda A, scale: fit_lr(A, y, 0.01, scale).predict_proba(A)[:, 1]
    np.testing.assert_allclose(probs(X, True), probs(1000 * X, True), atol=1e-3)
    assert np.abs(probs(X, False) - probs(1000 * X, False)).max() > 0.01


def test_fit_lr_c_reaches_the_model():
    X, y = features()
    assert np.linalg.norm(fit_lr(X, y, 1e-4, True)[-1].coef_) < np.linalg.norm(fit_lr(X, y, 1e2, True)[-1].coef_)


# --- bookkeeping: harness on value-coded features ----------------------------------------------------------------

CODE = {"hmp2": 2.0, "hv": 3.0}


def fake_feats(model=None):
    """AGP column 0 = row number + 10; every test-cohort row carries its cohort's code."""
    n = 200
    y = (np.arange(n) % 5 == 0).astype(int)
    X = np.c_[np.arange(n, dtype=np.float32) + 10, np.zeros(n, np.float32)]
    out = {"agp_ibd": (X, y)}
    for c, code in CODE.items():
        out[c] = (np.full((20, 2), code, np.float32), np.arange(20) % 2)
    return out


@pytest.fixture
def fakes(monkeypatch):
    """body_features -> fake_feats; train_head only records; predict_ibd returns column 0, so scores show their rows."""
    fits = []

    def fake_train(X, y, cfg, seed, on_epoch=None):
        fits.append((X.copy(), y.copy(), seed))
        for epoch in range(cfg.epochs + 1):
            on_epoch(epoch, None)

    monkeypatch.setattr(e5b_head, "body_features", fake_feats)
    monkeypatch.setattr(e5b_head, "train_head", fake_train)
    monkeypatch.setattr(e5b_head, "predict_ibd", lambda head, X: X[:, 0].copy())
    return fits


def test_config_matches_the_shipped_checkpoints():
    import json
    config = e5b_head.pm.path("ensemble_dir") / "run1_epoch0_disc" / "config.json"
    if not config.exists():
        pytest.skip("checkpoints not downloaded")
    c = json.loads(config.read_text())
    assert (POPE.dropout, POPE.init_std) == (c["hidden_dropout_prob"], c["initializer_range"])


def test_agp_split_is_stratified_disjoint_and_fixed():
    y = (np.arange(1000) % 20 == 0).astype(int)
    tr, ho = e5b_head.agp_split(y)
    assert sorted(np.r_[tr, ho]) == list(range(1000))
    assert len(ho) == 200 and y[ho].sum() == 10 and y[tr].sum() == 40
    np.testing.assert_array_equal(ho, e5b_head.agp_split(y)[1])


def test_head_cell_80_keeps_the_holdout_out_of_training(fakes):
    scores = e5b_head.head_cell("ens1", replace(POPE, epochs=3), 2, "80")
    (X_tr, y_tr, seed), = fakes
    assert seed == e5b_head.run_seed("ens1", 2)
    trained = set(X_tr[:, 0].astype(int))
    held = set(scores[1][e5b_head.HOLDOUT].astype(int))
    assert len(trained) == 160 and len(held) == 40
    assert not trained & held and trained | held == set(range(10, 210))
    np.testing.assert_array_equal(y_tr, (X_tr[:, 0].astype(int) - 10) % 5 == 0)     # labels still match their rows


def test_head_cell_scores_the_right_cohorts_at_the_checkpoints_only(fakes):
    scores = e5b_head.head_cell("ens1", replace(POPE, epochs=6), 0, "80")
    assert sorted(scores) == [e for e in E5B["checkpoints"] if e <= 6]
    for sets in scores.values():
        assert set(sets) == {"hmp2", "hv", e5b_head.HOLDOUT}
        assert (sets["hmp2"] == 2).all() and (sets["hv"] == 3).all()


def test_head_cell_all_trains_on_every_row_without_a_holdout(fakes):
    scores = e5b_head.head_cell("ens1", replace(POPE, epochs=1), 0, "all")
    assert len(fakes[0][0]) == 200
    assert set(scores[1]) == {"hmp2", "hv"}


def test_run_seed_differs_for_every_model_and_seed():
    seeds = [e5b_head.run_seed(m, s) for m in e5b_head.ALL_MODELS for s in E5B["seeds"]]
    assert len(set(seeds)) == len(seeds)


def test_lr_cell_fits_on_the_80_and_scores_the_holdout(monkeypatch):
    fits = []

    class Column0:
        def predict_proba(self, X):
            return np.c_[1 - X[:, 0], X[:, 0]]

    def fake_fit(X, y, C, scale):
        fits.append((set(X[:, 0].astype(int)), C, scale))
        return Column0()

    monkeypatch.setattr(e5b_head, "body_features", fake_feats)
    monkeypatch.setattr(e5b_head, "fit_lr", fake_fit)
    out = e5b_head.lr_cell("ens1", 0.5, False)
    (trained, C, scale), = fits
    assert (C, scale) == (0.5, False) and len(trained) == 160
    assert not trained & set(out[e5b_head.HOLDOUT].astype(int))
    assert (out["hv"] == 3).all()


def fake_runs(seeds=(0, 1), epochs=(0, 1)):
    rng = np.random.default_rng(0)
    return {(m, s): {e: {"hmp2": rng.random(20)} for e in epochs} for m in MODELS for s in seeds}


def test_ensemble_scores_average_the_members_of_one_seed():
    runs = fake_runs()
    ens = e5b_head.ensemble_scores(runs)
    assert sorted(ens) == [0, 1]
    np.testing.assert_allclose(ens[1][0]["hmp2"], np.mean([runs[m, 1][0]["hmp2"] for m in MODELS], axis=0))


def test_ensemble_scores_need_every_member():
    runs = fake_runs()
    del runs["ens3", 1]
    assert e5b_head.ensemble_scores(runs) == {}


def test_head_rows_count_and_values():
    runs, cfg = fake_runs(), replace(POPE, batch_size=32)
    labels = {"hmp2": np.arange(20) % 2}
    ens = e5b_head.ensemble_scores(runs)
    d = pd.DataFrame(e5b_head.head_rows("pope", "80", runs, ens, cfg, 100, labels))
    assert len(d) == 10 * 2 * 2 + 2 * 2                      # members x seeds x epochs, plus the ensembles
    r = d[(d.model == "ensemble") & (d.seed == 1) & (d.epoch == 1)].iloc[0]
    assert r.auroc == pytest.approx(roc_auc_score(labels["hmp2"], ens[1][1]["hmp2"]), abs=1e-4)
    assert r.steps == math.ceil(100 / 32) and r.test_set == "HMP2"
    assert r.auroc_ci_lo < r.auroc < r.auroc_ci_hi           # epoch 1 is a CI epoch
    assert d[(d.model == "ensemble") & (d.epoch == 0)].auroc_ci_lo.eq("").all()


def test_merge_csv_replaces_only_a_rerun_config(tmp_path):
    path, cols = tmp_path / "x.csv", ["config", "train_on", "auroc"]
    e5b_head.merge_csv(path, [{"config": "pope", "train_on": "80", "auroc": 0.1},
                              {"config": "ce", "train_on": "80", "auroc": 0.2}], cols, ["config", "train_on"])
    e5b_head.merge_csv(path, [{"config": "pope", "train_on": "80", "auroc": 0.9}], cols, ["config", "train_on"])
    d = pd.read_csv(path).set_index("config").auroc
    assert d.to_dict() == {"ce": 0.2, "pope": 0.9}


def test_pope_head_probs_match_the_electra_head():
    torch.manual_seed(0)
    ref = ElectraClassificationHead(ElectraConfig(hidden_size=200, num_labels=2)).eval()
    state = {f"dense.{k}": v for k, v in ref.dense.state_dict().items()}
    state |= {f"out_proj.{k}": v for k, v in ref.out_proj.state_dict().items()}
    X = np.random.default_rng(0).normal(size=(8, 200)).astype(np.float32)
    with torch.no_grad():
        expected = torch.softmax(ref(torch.as_tensor(X)[:, None]), 1)[:, 1].numpy()
    np.testing.assert_allclose(e5b_head.pope_head_probs(state, X), expected, rtol=1e-6)


def test_gate_passes_within_the_tolerance_of_the_seed_mean():
    tol = e5b_head.pm.CFG["e4"]["gate_tolerance"]
    assert e5b_head.gate_passes([0.60, 0.70], 0.65 + tol / 2)
    assert not e5b_head.gate_passes([0.60, 0.70], 0.65 + 2 * tol)
