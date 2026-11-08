"""Tests for the read-to-ASV mapping in src/e8_metagenome_v4.py (run: pytest src/test_e8_metagenome_v4.py)."""
import numpy as np
import pytest

from e8_metagenome_v4 import build_index, candidates, em, map_pairs, revcomp

RNG = np.random.default_rng(0)


def random_seq(n):
    return "".join(RNG.choice(list("ACGT"), n))


@pytest.fixture(scope="module")
def vocab():
    a = random_seq(140)
    b = a[:100] + random_seq(40)               # shares its first 100 nt with a
    c = random_seq(140)
    return [a, b, c]


@pytest.fixture(scope="module")
def index(vocab):
    return build_index(vocab)


def test_revcomp():
    assert revcomp("AACGT") == "ACGTT"


def test_read_covering_a_unique_region_maps_to_one_asv(vocab, index):
    flank = random_seq(30)
    read = flank + vocab[2][:120]              # 30 nt of genome before the window, then the ASV
    assert candidates(read, index) == {2}
    assert candidates(revcomp(read), index) == {2}   # either strand


def test_read_in_the_shared_part_is_ambiguous(vocab, index):
    read = random_seq(60) + vocab[0][:90]      # only the 100 nt that a and b share
    assert candidates(read, index) == {0, 1}


def test_read_off_the_window_misses(index):
    assert candidates(random_seq(150), index) is None


def test_map_pairs_counts_and_mate_agreement(vocab):
    index = build_index(vocab)
    shared = random_seq(60) + vocab[0][:90]
    only_b = vocab[1][60:] + random_seq(70)
    pairs = [(shared, random_seq(150)),        # one mate hits: ambiguous {a, b}
             (shared, only_b),                 # mates agree on b
             (vocab[2][:140] + "A" * 10, vocab[0][10:140] + "C" * 20),   # mates disagree
             (random_seq(150), random_seq(150))]                         # no hit
    out = map_pairs(pairs, index)
    assert (out["pairs"], out["hits"], out["conflict"]) == (4, 3, 1)
    assert out["unique"] == {1: 1}
    assert out["ambiguous"] == {frozenset({0, 1}): 1}


def test_em_shares_ambiguous_reads_by_unique_support():
    theta = em({0: 30, 1: 10}, {frozenset({0, 1}): 40})
    assert theta[0] + theta[1] == pytest.approx(80)
    assert theta[0] / theta[1] == pytest.approx(3, rel=1e-3)   # the 40 shared reads follow the 3:1 unique split
    even = em({}, {frozenset({5, 6}): 10})
    assert even[5] == pytest.approx(5) and even[6] == pytest.approx(5)
