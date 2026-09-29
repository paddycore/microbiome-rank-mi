#!/bin/bash
# Step 8 full grid (decisions at S2, 2026-09-27). Resumable: completed runs are skipped.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ENC="ra pa rank_within clr rank_raw"
$PY src/evaluate.py --rf-variant spec       --encodings $ENC --seeds 0 1 2 3 4 --protocols cross cv holdout
$PY src/evaluate.py --rf-variant unweighted --encodings $ENC --seeds 0 1 2 3 4 --protocols cross holdout
echo "=== primary + sensitivity done $(date) ==="
$PY src/evaluate.py --rf-variant unweighted --encodings $ENC --seeds 0 1 2 3 4 --protocols cv
echo "=== ALL DONE $(date) ==="
