"""Tests for src/e5c_signal.py (run: pytest src/test_e5c_signal.py).

auroc_ci, delta_auroc_ci, crossfit_proba, crossfit_residual, stratified_auroc and spearman_ci are yours to write, and
these tests pin down how they behave. The bootstrap tests replay the documented draws, so they check exact numbers, and
the cross-fit tests rebuild the documented folds. Everything below "bookkeeping" checks the harness on fakes; the
alignment test needs the real data and is skipped without it.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import e5c_signal
from e5c_signal import auroc_ci, crossfit_proba, crossfit_residual, delta_auroc_ci, spearman_ci, stratified_auroc


def labelled(n=200, pos=0.3, effect=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos).astype(int)
    return y, effect * y + rng.normal(size=n)


def noise(n=120, k=60, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(n) < 0.4).astype(int), rng.normal(size=(n, k))


def replay(stat, n, n_boot, seed, y=None):
    """The documented draws: rng.integers(0, n, n) each, a draw with one class skipped (when y is given)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y is not None and y[i].min() == y[i].max():
            continue
        out.append(stat(i))
    return np.percentile(out, [2.5, 97.5])


# --- auroc_ci --------------------------------------------------------------------------------------------------

def test_auroc_ci_point_and_replayed_interval():
    y, s = labelled()
    auc, lo, hi = auroc_ci(y, s, 300, 7)
    assert auc == pytest.approx(roc_auc_score(y, s))
    np.testing.assert_allclose([lo, hi], replay(lambda i: roc_auc_score(y[i], s[i]), len(y), 300, 7, y))


def test_auroc_ci_skips_single_class_draws():
    # about a third of the draws hold no positive; skipping them (not redrawing) gives the replayed interval
    y, s = np.array([0, 0, 0, 0, 0, 0, 1, 1]), np.array([0, 3, 1, 4, 2, 5, 2.5, 3.5])
    _, lo, hi = auroc_ci(y, s, 200, 0)
    np.testing.assert_allclose([lo, hi], replay(lambda i: roc_auc_score(y[i], s[i]), 8, 200, 0, y))


def test_auroc_ci_returns_floats_and_follows_its_seed():
    y, s = labelled()
    a = auroc_ci(y, s, 200, 1)
    assert len(a) == 3 and all(isinstance(v, float) for v in a)
    assert tuple(a) == tuple(auroc_ci(y, s, 200, 1)) and tuple(a[1:]) != tuple(auroc_ci(y, s, 200, 2)[1:])


# --- delta_auroc_ci --------------------------------------------------------------------------------------------

def test_delta_auroc_ci_is_paired():
    y, a = labelled(seed=1)
    b = a + np.random.default_rng(2).normal(scale=0.7, size=len(y))
    d, lo, hi = delta_auroc_ci(y, a, b, 300, 3)
    assert d == pytest.approx(roc_auc_score(y, a) - roc_auc_score(y, b))
    paired = replay(lambda i: roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]), len(y), 300, 3, y)
    np.testing.assert_allclose([lo, hi], paired)


def test_delta_auroc_ci_of_a_score_with_itself_is_zero():
    y, a = labelled()
    assert tuple(delta_auroc_ci(y, a, a, 100, 0)) == (0.0, 0.0, 0.0)


# --- crossfit_proba --------------------------------------------------------------------------------------------

def test_crossfit_proba_is_out_of_fold():
    # 60 noise columns on 120 rows: scored on its own training rows the model looks good, out of fold it must not
    y, Z = noise()
    in_sample = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Z, y).predict_proba(Z)[:, 1]
    assert roc_auc_score(y, in_sample) > 0.9
    p = crossfit_proba(y, Z, 5, 0)
    assert p.shape == (120,) and ((p > 0) & (p < 1)).all()
    assert roc_auc_score(y, p) < 0.7


@pytest.mark.parametrize("n_folds, seed", [(5, 4), (3, 11)])
def test_crossfit_proba_matches_the_documented_folds_and_model(n_folds, seed):
    y, Z = noise(k=3, seed=1)
    Z[:, 0] = 1000 * Z[:, 0] + 1.5 * y                        # a large column: only a scaler keeps the fit unchanged
    expected = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(Z, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Z[tr], y[tr])
        expected[te] = model.predict_proba(Z[te])[:, 1]
    np.testing.assert_allclose(crossfit_proba(y, Z, n_folds, seed), expected, rtol=1e-6)


# --- crossfit_residual -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("n_folds, seed", [(5, 6), (4, 2)])
def test_crossfit_residual_matches_the_documented_folds(n_folds, seed):
    y, Z = noise(k=2, seed=2)
    s = 2 * Z[:, 0] - Z[:, 1] + 0.3 * y + np.random.default_rng(3).normal(scale=0.1, size=len(y))
    expected = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(Z, y):
        expected[te] = s[te] - LinearRegression().fit(Z[tr], s[tr]).predict(Z[te])
    np.testing.assert_allclose(crossfit_residual(y, s, Z, n_folds, seed), expected, rtol=1e-6, atol=1e-9)


def test_crossfit_residual_is_out_of_fold():
    y, Z = noise(k=80)
    s = np.random.default_rng(5).normal(size=len(y))
    in_sample = s - LinearRegression().fit(Z, s).predict(Z)   # 80 columns soak up most of s on their own rows
    assert crossfit_residual(y, s, Z, 5, 0).std() > 2 * in_sample.std()


# --- stratified_auroc ------------------------------------------------------------------------------------------

def test_stratified_auroc_bins_by_rank_with_stable_ties():
    strata = np.array([5, 5, 5, 5, 1, 1, 9, 9, 9])
    y = np.array([0, 1, 0, 1, 0, 1, 1, 1, 1])
    s = np.array([.1, .9, .8, .2, .3, .7, .5, .5, .5])
    out = stratified_auroc(y, s, strata, 3, 50, 0)            # stable order: rows [4, 5, 0], [1, 2, 3], [6, 7, 8]
    assert [(t["bin"], t["n"], t["n_pos"], t["lo_strata"], t["hi_strata"]) for t in out] == \
        [(0, 3, 1, 1, 5), (1, 3, 2, 5, 5), (2, 3, 3, 9, 9)]
    for t, rows in zip(out[:2], ([4, 5, 0], [1, 2, 3])):
        assert (t["auroc"], t["lo"], t["hi"]) == pytest.approx(auroc_ci(y[rows], s[rows], 50, 0))
    assert out[0]["auroc"] == 1.0 and out[1]["auroc"] == 0.5
    assert all(np.isnan(out[2][k]) for k in ("auroc", "lo", "hi"))  # one class only: no AUROC, no crash


def test_stratified_auroc_keeps_tied_rows_in_their_original_order():
    # 63 tied rows: a stable sort keeps rows 0-20, 21-41, 42-62 together; the positives below fingerprint each bin
    y = np.zeros(63, int)
    y[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]] = 1
    y[[22, 30, 40]] = 1
    y[[42, 45, 50, 55, 58, 60, 62]] = 1
    out = stratified_auroc(y, np.arange(63.0), np.zeros(63), 3, 20, 0)
    assert [t["n_pos"] for t in out] == [11, 3, 7]


def test_stratified_auroc_intervals_come_from_auroc_ci_with_the_same_seed():
    y, s = labelled(n=150, seed=4)
    strata = np.random.default_rng(5).normal(size=150)
    out = stratified_auroc(y, s, strata, 3, 200, 9)
    for t, rows in zip(out, np.array_split(np.argsort(strata, kind="stable"), 3)):
        assert (t["auroc"], t["lo"], t["hi"]) == pytest.approx(auroc_ci(y[rows], s[rows], 200, 9))


@pytest.mark.parametrize("n, n_bins, sizes", [(10, 4, [3, 3, 2, 2]), (9, 3, [3, 3, 3])])
def test_stratified_auroc_bin_sizes(n, n_bins, sizes):
    y = np.arange(n) % 2
    out = stratified_auroc(y, np.arange(n, dtype=float), np.arange(n)[::-1].astype(float), n_bins, 20, 0)
    assert [t["n"] for t in out] == sizes and [t["bin"] for t in out] == list(range(n_bins))


# --- spearman_ci -----------------------------------------------------------------------------------------------

def test_spearman_ci_point_and_replayed_interval():
    rng = np.random.default_rng(8)
    a = rng.normal(size=150)
    b = a + rng.normal(size=150)
    rho, lo, hi = spearman_ci(a, b, 300, 9)
    assert rho == pytest.approx(spearmanr(a, b).statistic)
    np.testing.assert_allclose([lo, hi], replay(lambda i: spearmanr(a[i], b[i]).statistic, 150, 300, 9))


def test_spearman_ci_is_rank_based():
    rng = np.random.default_rng(10)
    a, b = rng.normal(size=80), rng.normal(size=80)
    assert spearman_ci(a, np.exp(b), 100, 0) == pytest.approx(spearman_ci(a, b, 100, 0))


# --- bookkeeping: harness on fakes -----------------------------------------------------------------------------

def test_sc1_one_fits_the_nested_models_on_the_right_columns(monkeypatch):
    # fakes: a "prediction" is the last column it was given, an "AUROC" is the mean of what it scores
    monkeypatch.setattr(e5c_signal, "crossfit_proba", lambda y, Z, f, s: Z[:, -1].astype(float))
    monkeypatch.setattr(e5c_signal, "crossfit_residual", lambda y, sc, Z, f, s: sc - Z[:, 0])
    monkeypatch.setattr(e5c_signal, "auroc_ci", lambda y, p, b, s: (float(np.mean(p)), 0.0, 1.0))
    monkeypatch.setattr(e5c_signal, "delta_auroc_ci", lambda y, a, b_, b, s: (float(np.mean(a) - np.mean(b_)), -1.0, 1.0))
    y = np.arange(20) % 2
    Z = np.c_[np.full(20, 2.0), np.full(20, 3.0)]
    score = np.full(20, 7.0)
    out = {m: v for m, v, *_ in e5c_signal.sc1_one(y, Z, score, [score, score + 1])}
    assert (out["M0"], out["M1"], out["M2"]) == (3.0, 7.0, 7.0)  # M0 sees Z only, M1 gets the score last, M2 the score alone
    assert (out["M1-M0"], out["M1-M2"], out["residual"]) == (4.0, 0.0, 5.0)
    assert {"M1-M0 members", "residual members"} <= set(out)


def test_slope_per_doubling():
    tokens = np.array([16, 32, 64, 128])
    assert e5c_signal.slope_per_doubling(tokens, 0.5 + 0.02 * np.log2(tokens)) == pytest.approx(0.02)


def test_sc2_pairs_random_k_with_top_k_at_the_same_k(monkeypatch):
    monkeypatch.setitem(e5c_signal.E4, "top_k", [16, 32])
    monkeypatch.setitem(e5c_signal.E4, "random_k", [16, 32])
    rows = []
    for test in ("HMP2", "Halfvarson"):
        for k in (16, 32):
            rows.append({"condition": f"top{k}", "test_set": test, "auroc": 0.70, "mean_tokens": k})
            rows += [{"condition": f"rand{k}", "test_set": test, "auroc": a, "mean_tokens": k} for a in (0.5, 0.6)]
    calls = []

    def fake_delta(condition, cohort, base_condition):
        calls.append((condition, cohort, base_condition))
        return {"delta": -0.15, "lo": -0.2, "hi": -0.1}

    d = pd.DataFrame(e5c_signal.sc2(pd.DataFrame(rows), fake_delta))
    assert sorted(calls) == sorted((f"rand{k}", c, f"top{k}") for k in (16, 32) for c in ("hmp2", "hv"))
    r = d[(d.cohort == "HMP2") & (d.metric == "rand-K auroc") & (d.K == 16)].iloc[0]
    assert (r.value, r.lo, r.hi) == pytest.approx((0.55, 0.5, 0.6))
    assert (d.metric == "random-K minus top-K").sum() == 4


def test_merge_csv_replaces_only_the_sections_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(e5c_signal, "CSV_PATH", tmp_path / "e5c.csv")
    e5c_signal.merge_csv([e5c_signal.row("SC-0", "hmp2", "auroc", 0.1), e5c_signal.row("SC-1", "hmp2", "M0", 0.2)],
                         ["SC-0", "SC-1"])
    e5c_signal.merge_csv([e5c_signal.row("SC-0", "hmp2", "auroc", 0.9)], ["SC-0"])
    d = pd.read_csv(tmp_path / "e5c.csv")
    assert dict(zip(d.section, d.value)) == {"SC-0": 0.9, "SC-1": 0.2}


def test_meta_and_scores_are_row_aligned():
    if not (e5c_signal.RESULTS / "e5_meta.csv").exists() or not (e5c_signal.PRED_DIR / "e4_sorted_r0_hmp2.npz").exists():
        pytest.skip("results not available")
    meta = e5c_signal.load_meta()
    assert np.array_equal(np.minimum(meta.richness, 512), meta.n_taxa)
    assert (meta.richness_ids <= meta.richness).all()
    for c in e5c_signal.TEST_COHORTS:
        y = meta[meta.cohort == c].y.to_numpy()
        S = e5c_signal.scores(c, y)                           # asserts every label vector against e5_meta
        assert {"ensemble", "rf_ra", "rf_pa"} <= set(S)
        assert all(sc.shape == y.shape and all(m.shape == y.shape for m in mem) for sc, mem in S.values())
