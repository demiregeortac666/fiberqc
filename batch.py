"""
batch.py — run the multiverse across a folder of recordings (thin CLI over fiberqc).

Folder convention:
    <name>.ppd or <name>.csv          the recording
    <name>_events.npy or _events.csv  its event times

    python batch.py datasets/
"""

import argparse
import csv
import glob
import os
import sys

import fiberqc as fqc


def find_events(rec_path):
    stem = os.path.splitext(rec_path)[0]
    for ext in ("_events.npy", "_events.csv", "_events.txt"):
        if os.path.exists(stem + ext):
            return stem + ext
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--fs", type=float)
    ap.add_argument("--ppd-dir", default="../photometry_preprocessing")
    a = ap.parse_args()

    recs = []
    for pat in ("*.ppd", "*.csv"):
        recs += glob.glob(os.path.join(a.folder, pat))
    recs = sorted(r for r in recs if not r.endswith("_events.csv"))
    if not recs:
        sys.exit(f"No recordings in {a.folder}")

    rows = []
    for path in recs:
        name = os.path.basename(path)
        ev = find_events(path)
        try:
            rec = fqc.load(path, events=ev, fs=a.fs, ppd_dir=a.ppd_dir)
            if rec.events is None:
                mr = fqc.motion_robustness(rec)
                print(f"  {name:40s} (no events) CV={mr['cv']*100:.1f}%  -> {mr['verdict']}")
                rows.append({"dataset": name, "mode": "motion", "verdict": mr["verdict"]})
                continue
            res = fqc.multiverse(rec)
            s = res.summary()
            print(f"  {name:40s} {s['n_significant']}/{s['n_pipelines']}  "
                  f"t {s['t_min']}-{s['t_max']}  -> {s['verdict']}")
            rows.append({"dataset": name, "mode": "peri-event", **s})
        except Exception as e:
            print(f"  ERROR {name}: {e}")

    if rows:
        with open("batch_summary.csv", "w", newline="") as f:
            keys = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        print("\nsaved -> batch_summary.csv")


if __name__ == "__main__":
    main()
