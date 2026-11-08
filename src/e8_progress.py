"""Progress of the E8 mapping run (src/e8_metagenome_v4.py), with a bar per running sample and an ETA.

The NTFS drive is read through a user-space driver, so read offsets are not visible from outside. Progress is estimated
instead: finished samples are counted from the output folder; a running sample's share is the time since it started
(reconstructed from the log: the pool starts one sample per free worker) over its expected time (minutes per GB of
read-1 file, from the samples already finished).

Usage: python src/e8_progress.py               (one snapshot)
       python src/e8_progress.py --follow 30   (refresh every 30 s; Ctrl-C to quit)
"""
import argparse
import re
import statistics
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from e8_metagenome_v4 import samples

ROOT, OUT, LOG = Path("/Volumes/DataDiabetes"), Path.home() / "diabetes_work/e8", Path.home() / "diabetes_work/e8_run.log"
WIDTH = 30


def bar(frac, width=WIDTH):
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def running_samples():
    """Sample folders whose read-1 file a worker has open."""
    pids = subprocess.run(["pgrep", "-f", "e8_metagenome_v4.py"], capture_output=True, text=True).stdout.split()
    if not pids:
        return set()
    out = subprocess.run(["lsof", "-Fn", "-p", ",".join(pids)], capture_output=True, text=True).stdout
    return {Path(line[1:]).parent.name for line in out.splitlines() if line.startswith("n") and line.endswith("_1.fq.gz")}


def parse_log(text):
    """(start of the latest run, [(sample, finished at, minutes) for every sample processed in any run])."""
    start, done = None, []
    run_start = None
    for line in text.splitlines():
        m = re.match(r"--- resumed (\d{4}-\d\d-\d\d \d\d:\d\d)", line)
        if m:
            run_start = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M")
            start = run_start
        m = re.match(r"\[\d+/\d+\] (\S+): pairs .* \(([\d.]+) min, ([\d.]+) h so far\)", line)
        if m:
            at = run_start + timedelta(hours=float(m.group(3))) if run_start else None
            done.append((m.group(1), at, float(m.group(2))))
    return start, done


def snapshot():
    jobs = samples(ROOT)
    order = {s: i for i, (_, s, *_rest) in enumerate(jobs)}
    size = {s: r1.stat().st_size / 1e9 for _, s, _, r1, _ in jobs}
    finished = {p.name[:-len(".json.gz")] for p in OUT.glob("*/*.json.gz")}
    running = sorted(running_samples() - finished, key=order.get)
    start, log_done = parse_log(LOG.read_text() if LOG.exists() else "")
    rate = statistics.median([mins / size[s] for s, _, mins in log_done if s in size]) if log_done else None  # min per GB

    # the pool hands out samples in order: the first n_workers start with the run, then one more at every completion
    n_workers = max(len(running), 1)
    starts = [start] * n_workers + sorted(at for _, at, _ in log_done if at and start and at >= start)
    starts = starts[-len(running):] if running else []
    now = datetime.now()

    lines, partial = [], 0.0
    for s, began in zip(running, starts):
        expected = rate * size[s] if rate else None
        frac = min((now - began).total_seconds() / 60 / expected, 0.99) if (expected and began) else 0.0
        partial += frac
        left = f"~{max(expected - (now - began).total_seconds() / 60, 0):.0f} min left" if expected and began else ""
        lines.append(f"  {s.split('_')[0]:>8s} {re.search(r'T2DM_(\S+?)_', s).group(1):>8s}  [{bar(frac, 20)}] {frac:4.0%}  {left}")

    total = len(jobs)
    overall = (len(finished) + partial) / total
    todo = [s for _, s, *_ in jobs if s not in finished and s not in running]
    eta = ""
    if rate:
        remaining = sum(rate * size[s] for s in todo)
        remaining += sum(max(rate * size[s] - (now - b).total_seconds() / 60, 0) for s, b in zip(running, starts) if b)
        workers = max(len(running), 1)
        finish = now + timedelta(minutes=remaining / workers)
        eta = f"ETA ~{finish:%H:%M} (about {remaining / workers / 60:.1f} h)  ·  {rate:.1f} min per GB per worker"
    state = "running" if running else ("finished" if len(finished) == total else "NOT RUNNING")
    return "\n".join([f"E8 mapping  {now:%H:%M:%S}  [{state}]",
                      f"[{bar(overall)}] {overall:5.1%}   {len(finished)}/{total} samples done, {len(running)} running",
                      *lines, eta])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--follow", type=int, default=0, help="refresh every N seconds")
    args = ap.parse_args()
    while True:
        text = snapshot()
        if args.follow:
            print("\033[2J\033[H" + text, flush=True)
            time.sleep(args.follow)
        else:
            print(text)
            break


if __name__ == "__main__":
    main()
