"""Tests for fit_head and cell in src/e4b_head.py (run: pytest src/test_e4b_head.py).

The cell tests swap in fake features whose values encode where they came from (cohort x order), so they check
that each side of the 2x2 reads the right order, without needing the real cached features.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import e4b_head
from e4b_head import fit_head

CODE = {"agp_ibd": 10, "hmp2": 20, "hv": 30}           # feature value = cohort code + 1 if shuffled


@pytest.fixture
def fakes(monkeypatch):
    calls, fits = [], []

    def fake_load(cohort, order, seed=None):
        calls.append((cohort, order, seed if order == "shuffled" else None))
        cls = np.full((6, 2), float(CODE[cohort] + (order == "shuffled")))
        return cls, -cls, np.array([0, 1] * 3)

    class Traceable:  # P(IBD) = first feature value, so the output shows which test features were used
        def predict_proba(self, X):
            return np.c_[1 - X[:, 0], X[:, 0]]

    def fake_fit(X, y, class_weight):
        fits.append((X[0, 0], len(X), class_weight))
        return Traceable()

    monkeypatch.setattr(e4b_head, "load_features", fake_load)
    monkeypatch.setattr(e4b_head, "fit_head", fake_fit)
    return calls, fits


# --- fit_head --------------------------------------------------------------------------------------------------

def toy(n=200, shift=100.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 3)) + shift
    X[:, 0] += 2 * y                                    # feature 0 carries the label
    return X, y


@pytest.mark.parametrize("cw", ["balanced", None])
def test_fit_head_is_scaled_logistic_regression(cw):
    X, y = toy()
    clf = fit_head(X, y, cw)
    assert clf[-1].class_weight == cw
    # every other setting must be sklearn's default (positional arguments land in `penalty`, not `class_weight`)
    assert clf[-1].get_params() == LogisticRegression(class_weight=cw, max_iter=5000).get_params()
    np.testing.assert_allclose(clf[0].mean_, X.mean(0))  # scaler statistics from the training features
    p = clf.predict_proba(X)[:, 1]
    assert p.shape == (len(y),) and p[y == 1].mean() > p[y == 0].mean()


# --- cell: the 2x2 bookkeeping ---------------------------------------------------------------------------------

@pytest.mark.parametrize("tr, te", [("sorted", "sorted"), ("sorted", "shuffled"),
                                    ("shuffled", "sorted"), ("shuffled", "shuffled")])
def test_cell_reads_train_and_test_orders(fakes, tr, te):
    calls, fits = fakes
    out = e4b_head.cell(tr, te, "cls", "balanced", 3)
    assert fits == [(CODE["agp_ibd"] + (tr == "shuffled"), 6, "balanced")]   # fitted once, on AGP in train order
    assert set(out) == {"hmp2", "hv"}
    for cohort, (y, p) in out.items():
        assert (p == CODE[cohort] + (te == "shuffled")).all()                 # predicted on test cohort in test order
        assert (y == [0, 1] * 3).all()
    if "shuffled" in (tr, te):
        assert any(c[1] == "shuffled" and c[2] == 3 for c in calls)           # the seed reaches the shuffled loads


def test_cell_mean_feature_and_class_weight(fakes):
    _, fits = fakes
    out = e4b_head.cell("sorted", "sorted", "mean", None, 0)
    assert fits == [(-CODE["agp_ibd"], 6, None)]
    assert (out["hmp2"][1] == -CODE["hmp2"]).all()
