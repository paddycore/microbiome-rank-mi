"""Tests for bag_features in src/e4c_bag.py (run: pytest src/test_e4c_bag.py)."""
import numpy as np

from e4c_bag import bag_features

EMB = np.array([[1.0, 0.0],      # taxon 0 (a real taxon, not padding)
                [0.0, 1.0],      # taxon 1
                [2.0, 2.0],      # taxon 2
                [9.0, 9.0]])     # taxon 3: never present below
ARR = np.array([[[0, 30], [2, 10], [0, 0], [0, 0]],    # taxa 0 and 2, then padding
                [[1, 5], [2, 5], [0, 5], [0, 0]]],     # taxa 1, 2, 0
               dtype=float)


def test_unweighted_is_plain_mean():
    out = bag_features(ARR, EMB, weighted=False)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[0], [1.5, 1.0])          # mean of taxon 0 and taxon 2
    np.testing.assert_allclose(out[1], [1.0, 1.0])          # mean of taxa 1, 2, 0


def test_weighted_uses_read_counts():
    out = bag_features(ARR, EMB, weighted=True)
    np.testing.assert_allclose(out[0], [0.75 * 1 + 0.25 * 2, 0.75 * 0 + 0.25 * 2])  # 30 and 10 reads
    np.testing.assert_allclose(out[1], [1.0, 1.0])          # equal counts: same as the plain mean


def test_taxon_zero_is_kept_and_padding_dropped():
    one = np.array([[[0, 7], [0, 0], [0, 0], [0, 0]]], dtype=float)  # only taxon 0 is present
    np.testing.assert_allclose(bag_features(one, EMB, weighted=False)[0], [1.0, 0.0])
    np.testing.assert_allclose(bag_features(one, EMB, weighted=True)[0], [1.0, 0.0])


def test_input_not_modified():
    before = ARR.copy()
    bag_features(ARR, EMB, weighted=True)
    assert (ARR == before).all()
