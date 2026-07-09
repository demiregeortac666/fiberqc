"""
guppy_to_csv.py
---------------
GuPPy stores signal, control, and events in three separate CSVs. This merges the
signal + control into one two-channel CSV and copies the event times, in the
convention FP-Robust's batch runner expects.

Input  (a GuPPy sample_data_csv_N folder):
    Sample_Signal_Channel.csv    columns: timestamps, data, sampling_rate
    Sample_Control_Channel.csv   columns: timestamps, data, sampling_rate
    Sample_TTL.csv               column:  timestamps  (event times, seconds)

Output (into ./datasets/):
    guppy.csv          columns: time, signal, control
    guppy_events.csv   one column of event times

Usage:
    python guppy_to_csv.py /path/to/sample_data_csv_1
"""

import os
import sys
import numpy as np
import pandas as pd

OUTDIR = "datasets"


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python guppy_to_csv.py /path/to/sample_data_csv_folder")
    src = sys.argv[1]

    sig = pd.read_csv(os.path.join(src, "Sample_Signal_Channel.csv"))
    ctl = pd.read_csv(os.path.join(src, "Sample_Control_Channel.csv"))
    ttl = pd.read_csv(os.path.join(src, "Sample_TTL.csv"))

    # signal and control share the same timestamps; align on the shorter length
    n = min(len(sig), len(ctl))
    time   = sig["timestamps"].to_numpy(float)[:n]
    signal = sig["data"].to_numpy(float)[:n]
    control = ctl["data"].to_numpy(float)[:n]

    os.makedirs(OUTDIR, exist_ok=True)
    out = pd.DataFrame({"time": time, "signal": signal, "control": control})
    out_path = os.path.join(OUTDIR, "guppy.csv")
    out.to_csv(out_path, index=False)

    events = ttl["timestamps"].to_numpy(float)
    ev_path = os.path.join(OUTDIR, "guppy_events.csv")
    np.savetxt(ev_path, events, delimiter=",")

    fs = 1 / np.median(np.diff(time))
    print(f"wrote {out_path}  ({n} samples, ~{fs:.0f} Hz)")
    print(f"wrote {ev_path}  ({len(events)} events, "
          f"range {events.min():.0f}-{events.max():.0f} s)")


if __name__ == "__main__":
    main()
