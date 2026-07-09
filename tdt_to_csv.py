"""
tdt_to_csv.py
-------------
Convert a TDT tank (Dv-style demodulated streams) into FP-Robust's two-channel
CSV. This tank has photometry streams but NO event epocs, so we produce a
recording without events -- it will be analyzed with the event-free motion
robustness metric (see motion_check.py).

TDT convention: two of the Dv streams are 465 nm (signal) and 405 nm (isosbestic
control). We pick the first two by default but PRINT the choice so you can verify
and swap if needed (--signal / --control).

Usage:
    python tdt_to_csv.py /path/to/tank_folder guppy_artifact
    python tdt_to_csv.py /path/to/tank_folder guppy_artifact --signal Dv1A --control Dv2A
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tdt

OUTDIR = "datasets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tank")
    ap.add_argument("name")
    ap.add_argument("--signal")
    ap.add_argument("--control")
    a = ap.parse_args()

    block = tdt.read_block(a.tank)
    streams = block.streams
    # candidate photometry streams: the long ones (drop tiny stores like Fi1r)
    phot = [k for k, v in streams.items() if len(v.data) > 1000]

    sig_key = a.signal or (phot[0] if len(phot) > 0 else None)
    ctl_key = a.control or (phot[1] if len(phot) > 1 else None)
    if sig_key is None or ctl_key is None:
        sys.exit(f"Could not find two photometry streams. Available: {list(streams)}")

    print(f"signal  = {sig_key}  (fs={streams[sig_key].fs:.1f})")
    print(f"control = {ctl_key}  (fs={streams[ctl_key].fs:.1f})")
    print("  (if these look swapped, re-run with --signal / --control)")

    signal  = np.asarray(streams[sig_key].data, float)
    control = np.asarray(streams[ctl_key].data, float)
    n = min(len(signal), len(control))
    signal, control = signal[:n], control[:n]
    fs = float(streams[sig_key].fs)
    time = np.arange(n) / fs

    os.makedirs(OUTDIR, exist_ok=True)
    out_path = os.path.join(OUTDIR, f"{name}.csv" if (name := a.name) else "tdt.csv")
    pd.DataFrame({"time": time, "signal": signal, "control": control}).to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({n} samples, {fs:.0f} Hz, no events)")


if __name__ == "__main__":
    main()
