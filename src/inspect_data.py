"""Step 3 inspection (STOP POINT S1). Read-only: never writes to data/raw/.

Describes every file under data/raw/microbiomedata/ without assuming a layout.
Per the Dryad README, abundance data are coordinate-list arrays of shape
(n_samples, max_sample_size, 2) where [:, :, 0] = vocab ID and [:, :, 1] = abundance,
so the checks below infer padding, orientation, count-vs-relative and sparsity
from the arrays themselves.

Writes reports/s1_inspection.txt and reports/s1_inspection.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATA = RAW / "microbiomedata"
if (DATA / "microbiomedata").is_dir():  # the zip contains a top-level microbiomedata/ folder
    DATA = DATA / "microbiomedata"
OUT_TXT = ROOT / "reports" / "s1_inspection.txt"
OUT_JSON = ROOT / "reports" / "s1_inspection.json"
EXPECTED_VOCAB = 26726
CHUNK = 1000

lines = []
report = {}


def log(s=""):
    print(s)
    lines.append(s)


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def is_integral(a):
    a = np.asarray(a)
    return bool(np.all(np.isfinite(a)) and np.all(a == np.round(a)))


def describe_coo(path, arr):
    """Stats for an (n, L, 2) coordinate-list array, computed in chunks."""
    n, L, _ = arr.shape
    n_taxa, row_sums, pad_ids, zero_ab_ids = [], [], Counter(), Counter()
    id_min, id_max, ab_min, ab_max = np.inf, -np.inf, np.inf, -np.inf
    ab_integral, id_integral = True, True
    contiguous, sorted_desc, dup_ids = True, True, 0
    all_ids = set()
    first_row_example = None
    for s in range(0, n, CHUNK):
        blk = np.asarray(arr[s:s + CHUNK], dtype=np.float64)
        ids, ab = blk[:, :, 0], blk[:, :, 1]
        present = ab > 0
        k = present.sum(1)
        n_taxa.append(k)
        row_sums.append(np.where(present, ab, 0).sum(1))
        id_min, id_max = min(id_min, ids.min()), max(id_max, ids.max())
        ab_min, ab_max = min(ab_min, ab.min()), max(ab_max, ab.max())
        ab_integral &= is_integral(ab[present])
        id_integral &= is_integral(ids)
        # padding convention: which IDs sit where abundance == 0
        zero_ab_ids.update(ids[~present].astype(np.int64).tolist()[:200000])
        # entries present only in a prefix of each row?
        contiguous &= bool(np.all(present == (np.arange(L)[None, :] < k[:, None])))
        # abundances already in descending order within the present prefix?
        for i in range(blk.shape[0]):
            a = ab[i, :k[i]]
            if sorted_desc and np.any(np.diff(a) > 0):
                sorted_desc = False
            u = ids[i, :k[i]]
            if np.unique(u).size != u.size:
                dup_ids += 1
            all_ids.update(u.astype(np.int64).tolist())
        if first_row_example is None:
            k0 = int(k[0])
            first_row_example = {
                "n_present": k0,
                "first5": blk[0, :5].tolist(),
                "last_present": blk[0, max(k0 - 1, 0)].tolist(),
                "first_padding": blk[0, k0].tolist() if k0 < L else None,
            }
    n_taxa = np.concatenate(n_taxa)
    row_sums = np.concatenate(row_sums)
    d = {
        "n_samples": int(n), "max_sample_size": int(L),
        "id_min": float(id_min), "id_max": float(id_max), "ids_integral": id_integral,
        "abund_min": float(ab_min), "abund_max": float(ab_max),
        "abund_integral_where_present": ab_integral,
        "taxa_per_sample_min_med_max": [int(n_taxa.min()), float(np.median(n_taxa)), int(n_taxa.max())],
        "n_samples_zero_taxa": int((n_taxa == 0).sum()),
        "n_samples_hitting_L": int((n_taxa == L).sum()),
        "row_sum_min_med_max": [float(row_sums.min()), float(np.median(row_sums)), float(row_sums.max())],
        "present_entries_are_prefix": contiguous,
        "present_entries_sorted_desc": sorted_desc,
        "samples_with_duplicate_ids": dup_ids,
        "ids_at_zero_abundance_top5": zero_ab_ids.most_common(5),
        "n_distinct_ids": len(all_ids),
        "dense_sparsity_vs_26726": float(1 - n_taxa.mean() / EXPECTED_VOCAB),
        "first_row_example": first_row_example,
    }
    return d, all_ids


def main():
    if not DATA.exists():
        sys.exit(f"{DATA} not found - unzip microbiomedata.zip first (Step 2)")

    log("=" * 78)
    log("S1 INSPECTION - file inventory")
    log("=" * 78)
    inventory = []
    for p in sorted(DATA.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            rel = p.relative_to(DATA)
            fmt = p.suffix.lstrip(".") or "no-extension"
            inventory.append({"file": str(rel), "format": fmt, "bytes": p.stat().st_size})
            log(f"{str(rel):55s} {fmt:13s} {fmt_bytes(p.stat().st_size):>10s}")
    report["inventory"] = inventory

    for readme in [p for p in DATA.rglob("*") if p.is_file() and p.name.upper().startswith("README")]:
        log()
        log(f"--- {readme.relative_to(DATA)} (verbatim) ---")
        log(readme.read_text(errors="replace"))

    log()
    log("=" * 78)
    log("S1 INSPECTION - arrays")
    log("=" * 78)
    arrays, id_sets = {}, {}
    for p in sorted(DATA.rglob("*.npy")):
        rel = str(p.relative_to(DATA))
        try:
            a = np.load(p, mmap_mode="r", allow_pickle=False)
        except ValueError as e:  # object arrays need pickle; report rather than guess
            log(f"\n{rel}: cannot mmap without pickle ({e}); loading with allow_pickle=True")
            a = np.load(p, allow_pickle=True)
        info = {"shape": list(a.shape), "dtype": str(a.dtype)}
        log(f"\n{rel}: shape={a.shape} dtype={a.dtype}")
        if a.ndim == 3 and a.shape[-1] == 2:
            d, ids = describe_coo(p, a)
            info.update(d)
            id_sets[rel] = ids
            for k, v in d.items():
                log(f"    {k}: {v}")
        elif a.ndim <= 2 and a.size <= 5_000_000:
            vals, counts = np.unique(np.asarray(a), return_counts=True)
            if vals.size <= 20:
                info["unique_values_verbatim"] = {repr(v.item() if hasattr(v, 'item') else v): int(c)
                                                  for v, c in zip(vals, counts)}
                log(f"    unique values (verbatim repr -> count): {info['unique_values_verbatim']}")
            else:
                info["n_unique"] = int(vals.size)
                info["min_max"] = [float(np.min(a)), float(np.max(a))]
                log(f"    {vals.size} unique values; min={np.min(a)} max={np.max(a)}")
        arrays[rel] = info
    report["arrays"] = arrays

    log()
    log("=" * 78)
    log("S1 INSPECTION - cross-file checks")
    log("=" * 78)

    def n_of(name):
        return arrays.get(name, {}).get("shape", [None])[0]

    # sample-count pairing between data and label files
    for data_f, label_f in [("IBD_train_otu.npy", "IBD_train_label.npy"),
                            ("IBD_test_otu.npy", "IBD_test_label.npy"),
                            ("total_IBD_otu.npy", "total_IBD_label.npy"),
                            ("schirmer_IBD_otu.npy", "schirmer_IBD_labels.npy"),
                            ("halfvarson_otu.npy", "halfvarson_IBD_labels.npy"),
                            ("fruitdata/FRUIT_FREQUENCY_all_otu.npy", "fruitdata/FRUIT_FREQUENCY_all_label.npy"),
                            ("vegdata/VEGETABLE_FREQUENCY_all_otu.npy", "vegdata/VEGETABLE_FREQUENCY_all_label.npy")]:
        log(f"{data_f} n={n_of(data_f)}  |  {label_f} n={n_of(label_f)}")
    tr, te, tot = n_of("IBD_train_otu.npy"), n_of("IBD_test_otu.npy"), n_of("total_IBD_otu.npy")
    if None not in (tr, te, tot):
        log(f"IBD train+test = {tr + te}  vs total_IBD = {tot}")
    etr, ete = n_of("train_encodings_1897.npy"), n_of("test_encodings_1897.npy")
    if None not in (etr, ete):
        log(f"AGP pretraining pool train+test = {etr + ete} (expected 18,480)")

    # is total_IBD literally train ++ test?
    try:
        tot_a = np.load(DATA / "total_IBD_otu.npy", mmap_mode="r")
        tr_a = np.load(DATA / "IBD_train_otu.npy", mmap_mode="r")
        te_a = np.load(DATA / "IBD_test_otu.npy", mmap_mode="r")
        if tot_a.shape[1:] == tr_a.shape[1:] == te_a.shape[1:]:
            same = (np.array_equal(tot_a[:tr_a.shape[0]], tr_a)
                    and np.array_equal(tot_a[tr_a.shape[0]:], te_a))
            log(f"total_IBD_otu == concat(IBD_train_otu, IBD_test_otu) in order: {same}")
            report["total_is_train_concat_test"] = same
        else:
            log(f"total/train/test max_sample_size differ: {tot_a.shape[1]}, {tr_a.shape[1]}, {te_a.shape[1]}")
    except FileNotFoundError as e:
        log(f"skip total-vs-concat check: {e}")

    # ID-match overlap preview (Step 4 computes the canonical figure)
    agp_key = "total_IBD_otu.npy" if "total_IBD_otu.npy" in id_sets else None
    if agp_key:
        agp_ids = id_sets[agp_key]
        for k in ("schirmer_IBD_otu.npy", "halfvarson_otu.npy"):
            if k in id_sets:
                s = id_sets[k]
                frac = len(s & agp_ids) / len(s) if s else float("nan")
                log(f"ID-match overlap preview: {k}: {len(s)} distinct IDs, "
                    f"{len(s & agp_ids)} also in {agp_key} -> {frac:.3f}")
                report.setdefault("overlap_preview", {})[k] = frac
        pool = set()
        for k in ("train_encodings_1897.npy", "test_encodings_1897.npy"):
            pool |= id_sets.get(k, set())
        if pool:
            log(f"distinct IDs in AGP pretraining pool (18,480): {len(pool)}")
            for k in ("schirmer_IBD_otu.npy", "halfvarson_otu.npy"):
                if k in id_sets:
                    s = id_sets[k]
                    log(f"    {k} IDs also in pretraining pool: {len(s & pool) / len(s):.3f}")

    # vocab embeddings + fasta
    log()
    ve = RAW / "vocab_embeddings.npy"
    if ve.exists():
        v = np.load(ve, mmap_mode="r")
        report["vocab_embeddings"] = {"shape": list(v.shape), "dtype": str(v.dtype),
                                      "bytes": ve.stat().st_size}
        log(f"vocab_embeddings.npy: shape={v.shape} dtype={v.dtype} bytes={ve.stat().st_size} "
            f"(expected (26726, 100) float64 = 21,380,800 data bytes + header)")
        zero_rows = int((np.abs(np.asarray(v)).sum(1) == 0).sum())
        log(f"    all-zero rows: {zero_rows}")
        report["vocab_embeddings"]["zero_rows"] = zero_rows
    fa = RAW / "seqs_.07_embed.fasta"
    if fa.exists():
        headers = [l[1:].strip() for l in fa.read_text().splitlines() if l.startswith(">")]
        report["fasta"] = {"n_records": len(headers), "first_headers": headers[:5]}
        log(f"seqs_.07_embed.fasta: {len(headers)} records; first headers: {headers[:5]}")

    OUT_TXT.write_text("\n".join(lines) + "\n")
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT_TXT} and {OUT_JSON}")


if __name__ == "__main__":
    main()
