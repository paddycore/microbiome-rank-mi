#!/bin/bash
# Re-run src/report.py when the sensitivity arm, and later the optional unweighted CV, finish.
cd "$(dirname "$0")/.."
LOG=results/logs/full_grid.log
until grep -q "primary + sensitivity done" "$LOG"; do sleep 60; done
.venv/bin/python src/report.py > results/logs/report_after_sensitivity.log 2>&1
until grep -q "ALL DONE" "$LOG"; do sleep 60; done
.venv/bin/python src/report.py > results/logs/report_after_all.log 2>&1
