"""Tests for src/e4_transforms.py (run: pytest src/test_e4_transforms.py).

Toy rows look like Pope's input: [CLS, taxa sorted by descending count..., PAD, PAD, ...].
Real rows are 513 wide; the transforms must work for any width.
"""
import numpy as np
import pytest

import e4_transforms as T
from tokens import CLS, PAD

P = PAD
IDS = np.array([[CLS, 10, 11, 12, 13, 14, P, P],
                [CLS, 20, 21, P, P, P, P, P],
                [CLS, 30, P, P, P, P, P, P]], dtype=np.int64)
COUNTS = np.array([[1, 50, 9, 9, 9, 2, 0, 0],      # one tie group: taxa 11, 12, 13
                   [1, 7, 7, 0, 0, 0, 0, 0],       # the whole row is one tie group
                   [1, 4, 0, 0, 0, 0, 0, 0]], dtype=float)
LONG = np.concatenate([[CLS], np.arange(100, 160), [P] * 10])[None].astype(np.int64)  # 60 taxa: shuffles show
PERMUTING = [lambda ids, rng: T.shuffle_tokens(ids, rng),
             lambda ids, rng: T.shuffle_within_ties(ids, COUNTS[: len(ids)], rng),
             lambda ids, rng: T.reverse_order(ids)]


def rng(seed=0):
    return np.random.default_rng(seed)


def real(row):
    return row[1:][row[1:] != PAD]


# --- rules every order transform must keep -------------------------------------------------------------------

@pytest.mark.parametrize("f", PERMUTING)
def test_cls_stays_first_and_padding_stays_put(f):
    out = f(IDS, rng())
    assert out.shape == IDS.shape and out.dtype == IDS.dtype
    assert (out[:, 0] == CLS).all()
    assert ((out == PAD) == (IDS == PAD)).all()


@pytest.mark.parametrize("f", PERMUTING)
def test_same_taxa_only_order_changes(f):
    out = f(IDS, rng())
    for a, b in zip(IDS, out):
        assert sorted(real(a)) == sorted(real(b))


@pytest.mark.parametrize("f", PERMUTING)
def test_input_not_modified(f):
    before = IDS.copy()
    f(IDS, rng())
    assert (IDS == before).all()


# --- shuffle_tokens --------------------------------------------------------------------------------------------

def test_shuffle_changes_order():
    assert not (T.shuffle_tokens(LONG, rng()) == LONG).all()


def test_shuffle_same_seed_same_result():
    assert (T.shuffle_tokens(LONG, rng(3)) == T.shuffle_tokens(LONG, rng(3))).all()


def test_shuffle_different_seeds_differ():
    assert not (T.shuffle_tokens(LONG, rng(1)) == T.shuffle_tokens(LONG, rng(2))).all()


# --- shuffle_within_ties ---------------------------------------------------------------------------------------

def test_ties_moves_only_tied_taxa():
    for seed in range(20):
        out = T.shuffle_within_ties(IDS, COUNTS, rng(seed))
        assert out[0, 1] == 10 and out[0, 5] == 14       # counts 50 and 2 are unique: they must not move
        assert sorted(out[0, 2:5]) == [11, 12, 13]
        assert out[2, 1] == 30


def test_ties_actually_shuffles_a_tie_group():
    seen = {tuple(T.shuffle_within_ties(IDS, COUNTS, rng(s))[0, 2:5]) for s in range(20)}
    assert len(seen) > 1


def test_ties_without_ties_is_identity():
    counts = np.concatenate([[1], np.arange(60, 0, -1), [0] * 10])[None].astype(float)  # all distinct
    assert (T.shuffle_within_ties(LONG, counts, rng()) == LONG).all()


# --- reverse_order ---------------------------------------------------------------------------------------------

def test_reverse_exact():
    expected = np.array([[CLS, 14, 13, 12, 11, 10, P, P],
                         [CLS, 21, 20, P, P, P, P, P],
                         [CLS, 30, P, P, P, P, P, P]])
    assert (T.reverse_order(IDS) == expected).all()


def test_reverse_twice_is_identity():
    assert (T.reverse_order(T.reverse_order(IDS)) == IDS).all()


# --- top_k -----------------------------------------------------------------------------------------------------

def test_top_k_exact():
    expected = np.array([[CLS, 10, 11, P, P, P, P, P],
                         [CLS, 20, 21, P, P, P, P, P],
                         [CLS, 30, P, P, P, P, P, P]])
    assert (T.top_k(IDS, 2) == expected).all()


def test_top_k_keeps_shape_and_input():
    before = IDS.copy()
    out = T.top_k(IDS, 1)
    assert out.shape == IDS.shape and (out[:, 0] == CLS).all() and (IDS == before).all()


def test_top_k_at_full_width_is_identity():
    assert (T.top_k(IDS, IDS.shape[1] - 1) == IDS).all()
    assert (T.top_k(IDS, 512) == IDS).all()
