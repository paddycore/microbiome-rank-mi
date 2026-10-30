"""Tests for partial_out, probe_score and selectivity in src/e5_probe.py (run: pytest src/test_e5_probe.py)."""
import numpy as np

import e5_probe
from e5_probe import partial_out, probe_score, selectivity


def blobs(n=300, k=3, sep=4.0, d=5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % k
    X = rng.normal(size=(n, d))
    X[:, 0] += sep * y
    return X, y


# --- partial_out -----------------------------------------------------------------------------------------------

def test_partial_out_fits_on_train_only():
    rng = np.random.default_rng(1)
    Ztr, Zte = rng.normal(size=(200, 2)), rng.normal(loc=3.0, size=(50, 2))   # test confounders shifted
    B, c = np.array([[1.0, -2.0, 0.5], [0.3, 0.0, 4.0]]), np.array([5.0, -1.0, 2.0])
    noise = 0.001 * rng.normal(size=(200, 3))
    delta = rng.normal(size=(50, 3))
    Rtr, Rte = partial_out(Ztr @ B + c + noise, Zte @ B + c + delta, Ztr, Zte)
    assert Rtr.shape == (200, 3) and Rte.shape == (50, 3)
    np.testing.assert_allclose(Rte, delta, atol=0.01)       # test rows use the TRAIN fit, so only delta is left
    np.testing.assert_allclose(Rtr.mean(0), 0, atol=1e-8)    # OLS with intercept: train residuals have mean 0
    np.testing.assert_allclose(Ztr.T @ Rtr, 0, atol=1e-8)    # ... and are orthogonal to the confounders


# --- probe_score -----------------------------------------------------------------------------------------------

def test_probe_score_finds_a_separable_signal():
    X, y = blobs()
    assert probe_score(X, y, C=1.0, seed=0) > 0.95


def test_probe_score_is_balanced_accuracy():
    # class 0 is 90% and fully separable; classes 1 and 2 look identical.
    # plain accuracy would be about 0.95, balanced accuracy about (1 + 0.5 + 0.5) / 3 = 0.67
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(540, int), np.ones(30, int), np.full(30, 2)]
    X = rng.normal(size=(600, 3))
    X[y == 0, 0] += 8
    assert 0.55 < probe_score(X, y, C=1.0, seed=0) < 0.8


def test_probe_score_partials_out_z():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(300, 2))
    y = (Z[:, 0] > 0).astype(int)                            # the label lives in the confounder only
    X = Z @ rng.normal(size=(2, 6)) + 0.05 * rng.normal(size=(300, 6))
    assert probe_score(X, y, C=1.0, seed=0) > 0.9
    assert probe_score(X, y, C=1.0, seed=0, Z=Z) < 0.65      # nothing left once Z is removed


def test_probe_score_scores_every_fold(monkeypatch):
    # one score per held-out fold, and (with Z) one partial_out per fold: catches a fit/score left outside the loop
    calls = {"score": 0, "partial": 0}
    real_score, real_partial = e5_probe.balanced_accuracy_score, e5_probe.partial_out

    def count_score(*a, **k):
        calls["score"] += 1
        return real_score(*a, **k)

    def count_partial(*a, **k):
        calls["partial"] += 1
        return real_partial(*a, **k)

    monkeypatch.setattr(e5_probe, "balanced_accuracy_score", count_score)
    monkeypatch.setattr(e5_probe, "partial_out", count_partial)
    X, y = blobs()
    e5_probe.probe_score(X, y, C=1.0, seed=0, Z=X[:, 1:3])
    assert calls == {"score": e5_probe.E5["n_folds"], "partial": e5_probe.E5["n_folds"]}


def test_probe_score_is_deterministic():
    X, y = blobs(sep=0.5)
    assert probe_score(X, y, C=0.1, seed=4) == probe_score(X, y, C=0.1, seed=4)


# --- selectivity -----------------------------------------------------------------------------------------------

def test_selectivity_high_for_real_signal_zero_for_none():
    X, y = blobs()
    s, c, d = selectivity(X, y, C=1.0, seed=0)
    assert s > 0.95 and c < 0.5 and np.isclose(d, s - c)
    rng = np.random.default_rng(5)
    _, _, d0 = selectivity(rng.normal(size=(300, 5)), y, C=1.0, seed=0)
    assert abs(d0) < 0.12
