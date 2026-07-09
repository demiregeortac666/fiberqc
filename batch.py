"""
batch.py
--------
Run the FP-Robust multiverse across a whole folder of recordings and summarize
which datasets are robust and which flip -- in one command.

Folder convention (put your recordings in one directory):
    <name>.ppd   or  <name>.csv        the recording (two channels: signal + control)
    <name>_events.npy or _events.csv   the event times for that recording

Each recording is matched to its own events file by name. Anything without a
matching events file is skipped with a note.

Usage:
    python batch.py <data_folder>
    python batch.py <data_folder> --fs 130      # fs for CSVs without a time column

Output:
    batch_summary.csv   one row per dataset: n_events, n_significant/16, t range, verdict
    prints the same table.
"""

import os
import sys
import csv
import glob
import argparse
import itertools

import numpy as np
from scipy.stats import ttest_1samp

from pipeline import preprocess, peri_event
from data_io import load_recording

AXES = {
    "low_pass":      [10.0, 2.0],
    "bleaching":     ["double_exp", "highpass"],
    "motion":        ["OLS", "robust"],
    "normalization": ["dFF", "zscore"],
}
BASELINE_WIN = (-1.0, 0.0)
RESPONSE_WIN = (0.0, 2.0)


def find_events(rec_path):
    """Find the events file that goes with a recording (<name>_events.npy/.csv)."""
    stem = os.path.splitext(rec_path)[0]
    for ext in ("_events.npy", "_events.csv", "_events.txt"):
        if os.path.exists(stem + ext):
            return stem + ext
    return None


def event_amplitudes(trace, time_s, events):
    t_win, W = peri_event(trace, time_s, events)
    base = W[:, (t_win >= BASELINE_WIN[0]) & (t_win < BASELINE_WIN[1])].mean(axis=1)
    resp = W[:, (t_win >= RESPONSE_WIN[0]) & (t_win < RESPONSE_WIN[1])].mean(axis=1)
    return resp - base


def run_one(signal, control, time_s, fs, events):
    keys = list(AXES)
    ts, n_sig = [], 0
    for combo in itertools.product(*AXES.values()):
        params = dict(zip(keys, combo))
        trace = preprocess(signal, control, time_s, fs, **params)
        amp = event_amplitudes(trace, time_s, events)
        t, p = ttest_1samp(amp, 0.0)
        ts.append(float(t))
        n_sig += p < 0.05
    n = len(ts)
    verdict = ("ROBUST" if n_sig == n else "NULL" if n_sig == 0 else "FLIP")
    return {"n_significant": int(n_sig), "n_pipelines": n,
            "t_min": round(min(ts), 2), "t_max": round(max(ts), 2),
            "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--fs", type=float)
    a = ap.parse_args()

    recs = []
    for pat in ("*.ppd", "*.csv"):
        recs += glob.glob(os.path.join(a.folder, pat))
    recs = sorted(r for r in recs if not r.endswith("_events.csv"))
    if not recs:
        sys.exit(f"No recordings found in {a.folder}")

    rows = []
    for rec_path in recs:
        name = os.path.basename(rec_path)
        ev = find_events(rec_path)
        if ev is None:
            print(f"  SKIP {name}: no matching _events file")
            continue
        try:
            r = load_recording(rec_path, ev, fs=a.fs)
            signal, control = r["signal"], r["control"]
            time_s, fs, events = r["time_s"], r["fs"], r["events"]
            if events.max() > time_s[-1] * 1.5:
                events = events / 1000
            res = run_one(signal, control, time_s, fs, events)
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            continue
        row = {"dataset": name, "n_events": len(events), **res}
        rows.append(row)
        print(f"  {name:45s} {res['n_significant']:2d}/16  "
              f"t {res['t_min']:.1f}-{res['t_max']:.1f}  -> {res['verdict']}")

    if not rows:
        sys.exit("No datasets processed.")

    with open("batch_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n---------------- BATCH SUMMARY ----------------")
    n = len(rows)
    robust = sum(r["verdict"] == "ROBUST" for r in rows)
    flip = sum(r["verdict"] == "FLIP" for r in rows)
    print(f"{n} datasets: {robust} robust, {flip} flip, {n - robust - flip} null.")
    print("saved -> batch_summary.csv")


if __name__ == "__main__":
    main()
