"""Step 5: abundance encodings.

Every encoding has the signature f(X_train, X_apply) -> X_apply_encoded (float32). Any statistic
(taxon medians, pseudo-count) is fitted on X_train only. Inputs are raw counts (samples x vocab).

  ra           p_ij = x_ij / sum_k x_ik
  pa           1[x_ij > 0]
  rank_within  rank of p_ij / m_j within sample i, descending (rank 1 = largest), m_j = median of the
               non-zero p_.j over X_train. Zeros -> 0. Taxa with no non-zero value in X_train have no m_j
               and are treated as absent (-> 0).
  clr          ln(p_ij + eps) - mean_k ln(p_ik + eps), eps = half the smallest non-zero p in X_train
  rank_raw     rank of p_ij within sample i, descending. Zeros -> 0.

Ranks: scipy.stats.rankdata(method='average') on the non-zero entries only, then floor.
Rank encodings also report ties per sample = number of non-zero entries whose ranked value is shared
with at least one other non-zero entry in the same sample.

This file name (required by the spec) shadows Python's stdlib `encodings` package, so import it with
`load_encodings()` below (by file path) rather than `import encodings`.
"""
import numpy as np
from scipy.stats import rankdata

CHUNK = 1000


def _relative(X):
    out = np.empty(X.shape, np.float32)
    for s in range(0, X.shape[0], CHUNK):
        b = np.asarray(X[s:s + CHUNK], dtype=np.float64)
        tot = b.sum(1, keepdims=True)
        out[s:s + CHUNK] = b / np.where(tot > 0, tot, 1)
    return out


def _rank_rows(V):
    """Descending within-row rank of the non-zero entries of V (float64), zeros -> 0. Returns (ranks, ties)."""
    R = np.zeros(V.shape, np.float32)
    ties = np.zeros(V.shape[0], np.int32)
    for i in range(V.shape[0]):
        nz = np.flatnonzero(V[i])
        if nz.size == 0:
            continue
        vals = V[i, nz]
        R[i, nz] = np.floor(rankdata(-vals, method="average"))
        _, inv, cnt = np.unique(vals, return_inverse=True, return_counts=True)
        ties[i] = int((cnt[inv] > 1).sum())
    return R, ties


def ra(X_train, X_apply):
    return _relative(X_apply)


def pa(X_train, X_apply):
    return (np.asarray(X_apply) > 0).astype(np.float32)


def taxon_medians(X_train):
    """m_j = median of the non-zero relative abundances of taxon j over X_train (NaN if never observed)."""
    P = _relative(X_train)
    m = np.full(P.shape[1], np.nan)
    for j in np.flatnonzero((P > 0).any(0)):
        col = P[:, j]
        m[j] = np.median(col[col > 0].astype(np.float64))
    return m


def rank_within(X_train, X_apply, return_ties=False):
    m = taxon_medians(X_train)
    seen = ~np.isnan(m)
    R = np.zeros(X_apply.shape, np.float32)
    ties = np.zeros(X_apply.shape[0], np.int32)
    for s in range(0, X_apply.shape[0], CHUNK):
        P = _relative(X_apply[s:s + CHUNK]).astype(np.float64)
        Q = np.zeros_like(P)
        Q[:, seen] = P[:, seen] / m[seen]
        R[s:s + CHUNK], ties[s:s + CHUNK] = _rank_rows(Q)
    return (R, ties) if return_ties else R


def clr(X_train, X_apply):
    P_train = _relative(X_train)
    eps = 0.5 * float(P_train[P_train > 0].min())
    del P_train
    out = np.empty(X_apply.shape, np.float32)
    for s in range(0, X_apply.shape[0], CHUNK):
        L = np.log(_relative(X_apply[s:s + CHUNK]).astype(np.float64) + eps)
        out[s:s + CHUNK] = L - L.mean(1, keepdims=True)
    return out


def rank_raw(X_train, X_apply, return_ties=False):
    R = np.zeros(X_apply.shape, np.float32)
    ties = np.zeros(X_apply.shape[0], np.int32)
    for s in range(0, X_apply.shape[0], CHUNK):
        R[s:s + CHUNK], ties[s:s + CHUNK] = _rank_rows(_relative(X_apply[s:s + CHUNK]).astype(np.float64))
    return (R, ties) if return_ties else R


ENCODINGS = {"ra": ra, "pa": pa, "rank_within": rank_within, "clr": clr, "rank_raw": rank_raw}
RANK_ENCODINGS = {"rank_within", "rank_raw"}


def encode(name, X_train, X_apply):
    """Returns (encoded X_apply, ties per sample or None)."""
    f = ENCODINGS[name]
    if name in RANK_ENCODINGS:
        return f(X_train, X_apply, return_ties=True)
    return f(X_train, X_apply), None
