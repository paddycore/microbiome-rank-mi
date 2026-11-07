"""Tests for pool() in src/e7_abundance_pool.py (run: pytest src/test_e7_abundance_pool.py)."""
import numpy as np

from e7_abundance_pool import pool


def toy():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(2, 5, 3))
    counts = np.array([[1, 6, 3, 1, 0],        # [CLS], three taxa, padding
                       [1, 2, 2, 0, 0]])
    return states, counts


def test_pool_weights_exclude_cls_and_padding():
    states, counts = toy()
    out = pool(states, counts, np.random.default_rng(1))
    np.testing.assert_allclose(out["cls"], states[:, 0])
    np.testing.assert_allclose(out["mean"][0], states[0, 1:4].mean(0))
    np.testing.assert_allclose(out["abund"][0], (6 * states[0, 1] + 3 * states[0, 2] + states[0, 3]) / 10)
    np.testing.assert_allclose(out["abund"][1], out["mean"][1])          # equal counts: abundance weights = presence


def test_pool_shuffled_keeps_the_weights_but_moves_them():
    states = np.zeros((1, 5, 3))
    states[0, 1:4] = np.eye(3)                 # one-hot token states expose the weight each taxon got
    counts = np.array([[1, 6, 3, 1, 0]])
    seen = {tuple(np.round(pool(states, counts, np.random.default_rng(s))["abund_shuffled"][0], 6)) for s in range(20)}
    assert all(sorted(w) == sorted([0.6, 0.3, 0.1]) for w in seen)    # same weights, assigned to different taxa
    assert len(seen) > 1
