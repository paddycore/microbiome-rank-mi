"""E8 step 1: turn the T2D shotgun metagenomes into V4 profiles over Pope's ASV vocabulary (notebook/plan_e8.md).

Pope's 26,726 ASVs are the 140 nt that follow the 515F primer. A read pair is mapped by exact 31-mers on both strands:
the candidate ASVs of a read are those consistent with EVERY 31-mer of it that the index contains, and both mates must
agree. Pairs with one candidate are counted directly; pairs with several are kept as ambiguous sets and shared out by
EM later (kallisto-style), so the analysis can compare unique-only, EM, and EM without 1-read ASVs.

Data stay outside the repo: the profiles are derived from patients. Output, one file per sample:
  <out>/<batch>/<sample>.json.gz   {"batch", "sample", "participant", "pairs", "hits", "conflict",
                                    "unique": {vocab id: n}, "ambiguous": [[n, id, id, ...], ...]}

Usage: caffeinate -i python src/e8_metagenome_v4.py --root /Volumes/DataDiabetes --out ~/diabetes_work/e8 --jobs 4
       (resumable: finished samples are skipped; about 10 min per sample per process)
"""
import argparse
import collections
import gzip
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import pope_model as pm

K = 31
PROBE_STRIDE = 8
COMPLEMENT = str.maketrans("ACGT", "TGCA")
INDEX = None                                       # set in the parent before forking, shared by the workers


def revcomp(s):
    return s.translate(COMPLEMENT)[::-1]


def load_vocab(path):
    """ASV sequences in vocab-ID order (FASTA record embed{j+1} = vocab ID j), without the trailing <unk> record."""
    seqs = [line.strip() for line in open(path) if not line.startswith(">")]
    return [s for s in seqs if len(s) >= K]


def build_index(seqs, k=K):
    """{k-mer: frozenset of vocab IDs}, from both strands of every sequence."""
    index = collections.defaultdict(set)
    for i, s in enumerate(seqs):
        for seq in (s, revcomp(s)):
            for p in range(len(seq) - k + 1):
                index[seq[p:p + k]].add(i)
    return {km: frozenset(v) for km, v in index.items()}


def candidates(read, index, k=K, stride=PROBE_STRIDE):
    """Vocab IDs consistent with every indexed k-mer of the read; None if the read misses the V4 window.

    A cheap probe first (every `stride`-th k-mer): a read that overlaps an ASV by at least k + stride - 1 nt is always
    caught; shorter overlaps are missed, which only costs yield.
    """
    if not any(read[p:p + k] in index for p in range(0, len(read) - k + 1, stride)):
        return None
    sets = [index[read[p:p + k]] for p in range(len(read) - k + 1) if read[p:p + k] in index]
    return frozenset.intersection(*sets)


def map_pairs(pairs, index):
    """Counts over an iterable of (read1, read2) strings."""
    out = {"pairs": 0, "hits": 0, "conflict": 0, "unique": collections.Counter(), "ambiguous": collections.Counter()}
    for r1, r2 in pairs:
        out["pairs"] += 1
        found = [c for c in (candidates(r1, index), candidates(r2, index)) if c is not None]
        if not found:
            continue
        out["hits"] += 1
        cand = frozenset.intersection(*found)
        if not cand:
            out["conflict"] += 1
        elif len(cand) == 1:
            out["unique"][next(iter(cand))] += 1
        else:
            out["ambiguous"][cand] += 1
    return out


def em(unique, ambiguous, iters=100):
    """{vocab id: reads} after sharing each ambiguous set among its members in proportion to their abundance.

    `unique` {id: n}, `ambiguous` {frozenset of ids: n}. Members with no unique support start from a tiny prior, so
    a set made only of unsupported ASVs is shared evenly.
    """
    theta = {a: float(c) for a, c in unique.items()}
    for _ in range(iters):
        new = collections.Counter({a: float(c) for a, c in unique.items()})
        for cand, n in ambiguous.items():
            w = [(a, theta.get(a, 0.0) + 1e-9) for a in cand]
            total = sum(x for _, x in w)
            for a, x in w:
                new[a] += n * x / total
        theta = new
    return dict(theta)


def fastq_pairs(r1, r2):
    with gzip.open(r1, "rt") as f1, gzip.open(r2, "rt") as f2:
        for i, (a, b) in enumerate(zip(f1, f2)):
            if i % 4 == 1:
                yield a.strip(), b.strip()


def samples(root):
    """(batch, sample, participant, read1, read2) for every WGS sample under the drive's dated batch folders."""
    out = []
    for d in sorted(Path(root).glob("20*_Diabetes/WGS/*")):
        r1, r2 = sorted(d.glob("*_1.fq.gz")), sorted(d.glob("*_2.fq.gz"))
        if r1 and r2:
            m = re.search(r"T2DM_(\d+-\d+)", d.name)
            out.append((d.parent.parent.name, d.name, m.group(1) if m else "", r1[0], r2[0]))
    return out


def run_one(job):
    batch, sample, participant, r1, r2, out_dir = job
    path = Path(out_dir) / batch / f"{sample}.json.gz"
    if path.exists():
        return sample, "skipped", 0.0
    t0 = time.time()
    res = map_pairs(fastq_pairs(r1, r2), INDEX)
    record = {"batch": batch, "sample": sample, "participant": participant,
              **{k: res[k] for k in ("pairs", "hits", "conflict")},
              "unique": {str(a): n for a, n in res["unique"].items()},
              "ambiguous": [[n, *sorted(c)] for c, n in res["ambiguous"].items()]}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(record, f)
    tmp.rename(path)
    return sample, f"pairs {res['pairs']:,} hits {res['hits']:,} unique {sum(res['unique'].values()):,}", time.time() - t0


def main():
    global INDEX
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N samples (for a trial run)")
    args = ap.parse_args()
    out_dir = Path(args.out).expanduser()
    jobs = [(*s, out_dir) for s in samples(args.root)][:args.limit]
    t0 = time.time()
    INDEX = build_index(load_vocab(pm.path("vocab_embeddings").with_name("seqs_.07_embed.fasta")))
    print(f"{len(jobs)} samples, index of {len(INDEX):,} {K}-mers built in {time.time() - t0:.0f}s", flush=True)
    with mp.get_context("fork").Pool(args.jobs) as pool:
        for i, (sample, msg, dt) in enumerate(pool.imap_unordered(run_one, jobs), 1):
            print(f"[{i}/{len(jobs)}] {sample}: {msg} ({dt / 60:.1f} min, {(time.time() - t0) / 3600:.2f} h so far)", flush=True)


if __name__ == "__main__":
    main()
