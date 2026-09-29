"""Verify Dryad downloads in data/raw/ against the sizes and SHA-256 digests
published by the Dryad API for dataset doi:10.5061/dryad.tb2rbp08p, version 9
(2025-02-20). Digests fetched from https://datadryad.org/api/v2/versions/346742/files
on 2026-09-27.

Usage: python src/verify_downloads.py [--move-from ~/Downloads]
  --move-from DIR  first move any matching files found in DIR into data/raw/
                   (README.md is renamed to README_dryad.md)
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# local name -> (dryad file id, dryad name, bytes, sha256)
EXPECTED = {
    "README_dryad.md": (3885177, "README.md", 24346,
                        "18902dcf5fd4f9e0bb895bdf082fbb457ede5b1071b9887751b74b3108475749"),
    "microbiomedata.zip": (3885157, "microbiomedata.zip", 103930914,
                           "8c4e241441a02abdcca02a0f3dc72585d82192b9f68abffde0b07a3aee9e97bd"),
    "vocab_embeddings.npy": (3885160, "vocab_embeddings.npy", 21380928,
                             "a4d624cb88d0fa2d181bed078bdc13e5b2c3dc030d204e57b7943e5c37a51906"),
    "seqs_.07_embed.fasta": (3885158, "seqs_.07_embed.fasta", 4131444,
                             "8778378adc892dbb55aa991ded204cf30877eba0367fbf87aa70664224e4e64b"),
}
DEFERRED = {  # not needed until STOP POINT S3
    "pretrainedmodels.zip": (3885159, "pretrainedmodels.zip", 146813682,
                             "07cfffa6751a5348ede77d65e35c05d70e75e1e2b994fad9b46202cbbf0aa2ba"),
    "ensemble.zip": (3885161, "ensemble.zip", 163173391,
                     "69c3b8d4e02d923aa686671d78ffc66be070b36588be79ea979da2daa75cac7a"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def move_from(src_dir: Path) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for local, (_, dryad_name, _, _) in {**EXPECTED, **DEFERRED}.items():
        dst = RAW / local
        if dst.exists():
            continue
        for cand in (src_dir / dryad_name, src_dir / local):
            if cand.exists():
                shutil.move(str(cand), dst)
                print(f"moved {cand} -> {dst}")
                break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--move-from", type=Path)
    args = ap.parse_args()
    if args.move_from:
        move_from(args.move_from.expanduser())

    ok = True
    for table, required in ((EXPECTED, True), (DEFERRED, False)):
        for local, (fid, _, size, digest) in table.items():
            p = RAW / local
            if not p.exists():
                if required:
                    ok = False
                    print(f"MISSING  {local}  (https://datadryad.org/downloads/file_stream/{fid})")
                continue
            actual_size = p.stat().st_size
            actual = sha256(p)
            good = actual_size == size and actual == digest
            ok &= good or not required
            print(f"{'OK      ' if good else 'MISMATCH'} {local}  size={actual_size} (expected {size})"
                  f"  sha256 {'matches' if actual == digest else 'DIFFERS: ' + actual}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
