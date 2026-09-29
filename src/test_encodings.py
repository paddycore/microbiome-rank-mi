"""Unit tests for src/encodings.py on a hand-computed 3x4 toy matrix.

Run: python -m unittest src/test_encodings.py -v

Toy counts (each row sums to 100, so p = x / 100):
    X = [[10,  0, 30, 60],
         [ 5,  5,  0, 90],
         [ 0, 20, 20, 60]]
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import load_encodings  # noqa: E402

enc = load_encodings()

X = np.array([[10, 0, 30, 60],
              [5, 5, 0, 90],
              [0, 20, 20, 60]], dtype=np.float32)
P = np.array([[.10, 0, .30, .60],
              [.05, .05, 0, .90],
              [0, .20, .20, .60]])


class TestEncodings(unittest.TestCase):
    def test_ra(self):
        np.testing.assert_allclose(enc.ra(X, X), P, rtol=1e-6)

    def test_ra_ignores_train(self):
        np.testing.assert_allclose(enc.ra(X[:1], X), P, rtol=1e-6)

    def test_pa(self):
        np.testing.assert_array_equal(enc.pa(X, X), [[1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1]])

    def test_rank_raw(self):
        # row0: .6->1 .3->2 .1->3 ; row1: .9->1, .05/.05 tie at 2,3 -> 2.5 -> floor 2 ; row2: .6->1, .2/.2 -> 2
        R, ties = enc.rank_raw(X, X, return_ties=True)
        np.testing.assert_array_equal(R, [[3, 0, 2, 1], [2, 2, 0, 1], [0, 2, 2, 1]])
        np.testing.assert_array_equal(ties, [0, 2, 2])

    def test_taxon_medians(self):
        # non-zero p per column: {.1,.05}->.075  {.05,.2}->.125  {.3,.2}->.25  {.6,.9,.6}->.6
        np.testing.assert_allclose(enc.taxon_medians(X), [.075, .125, .25, .6], rtol=1e-6)

    def test_rank_within(self):
        # q = p/m: row0 [1.333, 0, 1.2, 1.0] ; row1 [.667, .4, 0, 1.5] ; row2 [0, 1.6, .8, 1.0]
        R, ties = enc.rank_within(X, X, return_ties=True)
        np.testing.assert_array_equal(R, [[1, 0, 2, 3], [2, 3, 0, 1], [0, 1, 3, 2]])
        np.testing.assert_array_equal(ties, [0, 0, 0])

    def test_rank_within_uses_train_stats_only(self):
        # train = row0 only: m = [.1, undefined, .3, .6]. Apply to row1: q = [.5, (unseen -> 0), 0, 1.5]
        R = enc.rank_within(X[:1], X[1:2])
        np.testing.assert_array_equal(R, [[2, 0, 0, 1]])

    def test_clr(self):
        eps = 0.5 * 0.05  # smallest non-zero p in the training set
        expected = []
        for row in P:
            logs = [math.log(p + eps) for p in row]
            mu = sum(logs) / len(logs)
            expected.append([l - mu for l in logs])
        out = enc.clr(X, X)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out.sum(1), 0, atol=1e-5)

    def test_clr_eps_from_train_only(self):
        # train = row2 only: smallest non-zero p = .2 -> eps = .1, applied to row1
        eps = 0.1
        logs = [math.log(p + eps) for p in P[1]]
        mu = sum(logs) / 4
        np.testing.assert_allclose(enc.clr(X[2:3], X[1:2]), [[l - mu for l in logs]], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
