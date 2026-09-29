"""Step 4: build canonical artifacts in data/processed/ from the Dryad coordinate-list arrays.

Source arrays are (n_samples, 1897, 2): [:, :, 0] = vocab ID, [:, :, 1] = raw read count.
Padding is (0, 0) and ID 0 is also a real taxon, so an entry is present iff count > 0.

Two densifications are produced (decisions recorded at S1, see reports/phase0_log.md):

  canonical      counts, duplicate IDs within a sample SUMMED, taxon 0 kept. Used by every encoding.
  pope_faithful  emulates DeepMicro/data/convert_data.py used for Pope's No-DR baseline: positions are
                 written in order so the last write wins (duplicates overwritten, and the trailing padding
                 (0, 0) zeroes taxon 0), then rows are normalised to sum to 1 and rounded to 4 decimals.
                 Used only for the S2 diagnostic.

Vocabulary: the full 26,726-ID Pope/Tataru vocabulary (built from AGP; HMP2/HV were mapped onto it by the
data curators). Locked once, column j = vocab ID j = FASTA record embed{j+1}.
"""
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SRC = RAW / "microbiomedata" / "microbiomedata"
OUT = ROOT / "data" / "processed"
VOCAB_SIZE = 26726
CHUNK = 1000

COHORTS = {  # output name -> (data file, label file)
    "agp_ibd": ("total_IBD_otu.npy", "total_IBD_label.npy"),
    "hmp2": ("schirmer_IBD_otu.npy", "schirmer_IBD_labels.npy"),
    "hv": ("halfvarson_otu.npy", "halfvarson_IBD_labels.npy"),
}
# Pope et al. 2025 §3.2: BLAST, >=99% identity, e-value 1e-20, on the raw (pre-mapping) ASVs.
POPE_BLAST_OVERLAP = {"hmp2": 0.51, "hv": 0.34}


def densify_canonical(arr):
    """Counts matrix (n, VOCAB_SIZE) float32; duplicate IDs summed."""
    blocks = []
    for s in range(0, arr.shape[0], CHUNK):
        b = np.asarray(arr[s:s + CHUNK])
        r, c = np.nonzero(b[:, :, 1] > 0)
        m = sp.coo_matrix((b[r, c, 1], (r, b[r, c, 0].astype(np.int64))),
                          shape=(b.shape[0], VOCAB_SIZE)).tocsr()  # tocsr sums duplicates
        blocks.append(m.toarray().astype(np.float32))
    return np.vstack(blocks)


def densify_pope(arr):
    """Emulate convert_data.py: sequential assignment over all positions incl. padding, then RA rounded to 4 dp."""
    n, L, _ = arr.shape
    X = np.zeros((n, VOCAB_SIZE), np.float32)
    for s in range(0, n, CHUNK):
        b = np.asarray(arr[s:s + CHUNK])
        nb = b.shape[0]
        rows = np.repeat(np.arange(nb), L)
        ids = b[:, :, 0].astype(np.int64).ravel()
        vals = b[:, :, 1].ravel()
        # keep the LAST position written for each (row, id): reverse, then first occurrence of each key
        key = rows * VOCAB_SIZE + ids
        rev_key = key[::-1]
        _, first_in_rev = np.unique(rev_key, return_index=True)
        keep = len(key) - 1 - first_in_rev
        dense = np.zeros((nb, VOCAB_SIZE), np.float64)
        dense[rows[keep], ids[keep]] = vals[keep]
        tot = dense.sum(1, keepdims=True)
        dense /= np.where(tot > 0, tot, 1)
        X[s:s + nb] = np.around(dense, decimals=4)
    return X


def load_labels(name):
    y = np.load(SRC / name).reshape(-1)
    assert set(np.unique(y).tolist()) <= {0, 1}, f"unexpected labels in {name}: {np.unique(y)}"
    return y.astype(np.int8)


def distinct_ids(X):
    return set(np.flatnonzero((X > 0).any(0)).tolist())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "Dryad doi:10.5061/dryad.tb2rbp08p v9 (2025-02-20), microbiomedata.zip sha256 8c4e2414...97bd",
        "vocab": {"size": VOCAB_SIZE, "file": "vocab.txt",
                  "definition": "Pope/Tataru AGP vocabulary; column j = vocab ID j = FASTA header embed{j+1} "
                                "in seqs_.07_embed.fasta (embed26727 = <unk>, excluded)"},
        "matrix_values": "raw read counts (float32); relative abundance is derived in the encoding layer",
        "duplicate_ids": "summed within sample (canonical); overwritten last-wins (pope_faithful)",
        "cohorts": {},
    }
    (OUT / "vocab.txt").write_text("\n".join(str(j) for j in range(VOCAB_SIZE)) + "\n")

    agp_ids = None
    for name, (data_f, label_f) in COHORTS.items():
        arr = np.load(SRC / data_f, mmap_mode="r")
        y = load_labels(label_f)
        assert arr.shape[0] == y.shape[0]
        X = densify_canonical(arr)
        Xp = densify_pope(arr)
        np.save(OUT / f"{name}_X.npy", X)
        np.save(OUT / f"{name}_y.npy", y)
        np.save(OUT / f"{name}_X_popefaithful.npy", Xp)

        # how much the duplicate handling matters for this cohort
        present = np.asarray(arr[:, :, 1] > 0)
        n_entries = int(present.sum())
        n_cells = int((X > 0).sum())
        ids = distinct_ids(X)
        info = {
            "n_samples": int(X.shape[0]), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
            "pos_frac": round(float(y.mean()), 4),
            "sparsity": round(float(1 - n_cells / X.size), 5),
            "n_distinct_vocab_ids": len(ids),
            "read_depth_median": float(np.median(X.sum(1))),
            "entries_merged_as_duplicates": n_entries - n_cells,
            "pope_faithful_taxon0_nonzero_samples": int((Xp[:, 0] > 0).sum()),
        }
        if name == "agp_ibd":
            agp_ids = ids
        else:
            inter = ids & agp_ids
            in_agp = np.zeros(VOCAB_SIZE, bool)
            in_agp[list(agp_ids)] = True
            info["asv_overlap_frac_idmatch"] = round(len(inter) / len(ids), 4)
            info["asv_overlap_read_weighted_idmatch"] = round(float(X[:, in_agp].sum() / X.sum()), 4)
            info["asv_overlap_frac_pope_blast"] = POPE_BLAST_OVERLAP[name]
        manifest["cohorts"][name] = info
        print(name, json.dumps(info))
        del X, Xp

    manifest["asv_overlap_note"] = (
        "asv_overlap_frac_idmatch = fraction of the cohort's distinct vocab IDs that occur in AGP-IBD. It is near 1 by "
        "construction because the curators mapped HMP2/HV ASVs onto the AGP vocabulary and dropped non-matching ones; "
        "it does not measure shift. Fig 2 uses asv_overlap_frac_pope_blast (Pope et al. 2025 §3.2, BLAST >=99% id, "
        "e-value 1e-20, on raw ASVs) - their measurement, not ours (decision at S1, 2026-09-27).")
    manifest["subject_id"] = "none available; AGP-IBD 5-fold CV is stratified by label but NOT grouped by patient"
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", OUT / "manifest.json")


if __name__ == "__main__":
    main()
