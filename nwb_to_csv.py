"""
nwb_to_csv.py — convert NWB fiber photometry recordings (e.g. from the DANDI
archive) into the two-channel CSV that fiberqc reads.

    python nwb_to_csv.py datasets/dandi/001340 --outdir datasets/nwb

For each .nwb file it writes:
    <name>.csv          time, signal (470 nm), control (415/405 nm)
    <name>_events.csv   event times in seconds

Event times come from the NWB trials table. Choose which trials with --event-col
(default: side_in, the port entry) and --rewarded:

    --rewarded 1     only rewarded trials
    --rewarded 0     only unrewarded trials
    --rewarded all   every trial (default)

Tested against DANDI:001340 (Wilbrecht lab, NAc dLight, instrumental learning).
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")   # pynwb schema-version chatter

SIGNAL_KEYS  = ["470", "465", "signal", "dlight", "grab", "gcamp"]
CONTROL_KEYS = ["415", "405", "410", "iso", "control"]


def _pick_series(acq, hints):
    for h in hints:
        for k in acq:
            if h in k.lower():
                return k
    return None


def convert(path, outdir, event_col="side_in", rewarded="all"):
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(path, "r", load_namespaces=True) as io:
        nwb = io.read()

        sig_key = _pick_series(nwb.acquisition, SIGNAL_KEYS)
        ctl_key = _pick_series(nwb.acquisition, CONTROL_KEYS)
        if sig_key is None or ctl_key is None:
            raise ValueError(f"no signal/control series in {os.path.basename(path)}: "
                             f"{list(nwb.acquisition)}")

        sig_series = nwb.acquisition[sig_key]
        ctl_series = nwb.acquisition[ctl_key]
        signal = np.asarray(sig_series.data[:], float)
        control = np.asarray(ctl_series.data[:], float)

        if sig_series.timestamps is not None:
            t = np.asarray(sig_series.timestamps[:], float)
        else:
            t = sig_series.starting_time + np.arange(len(signal)) / sig_series.rate

        n = min(len(signal), len(control), len(t))
        signal, control, t = signal[:n], control[:n], t[:n]

        if nwb.trials is None:
            raise ValueError(f"no trials table in {os.path.basename(path)}")
        trials = nwb.trials.to_dataframe()
        if event_col not in trials.columns:
            raise ValueError(f"{event_col!r} not in trials: {list(trials.columns)}")

        if rewarded != "all":
            want = float(rewarded)
            trials = trials[trials["rewarded"] == want]
        events = trials[event_col].dropna().to_numpy(float)
        events = events[(events > t[0]) & (events < t[-1])]

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    if rewarded != "all":
        stem += "_rew" if rewarded == "1" else "_unrew"

    csv_path = os.path.join(outdir, f"{stem}.csv")
    pd.DataFrame({"time": t, "signal": signal, "control": control}).to_csv(csv_path, index=False)

    ev_path = os.path.join(outdir, f"{stem}_events.csv")
    np.savetxt(ev_path, events, delimiter=",")

    fs = 1 / np.median(np.diff(t))
    return csv_path, len(events), fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="an .nwb file, or a folder to search recursively")
    ap.add_argument("--outdir", default="datasets/nwb")
    ap.add_argument("--event-col", default="side_in")
    ap.add_argument("--rewarded", default="all", choices=["all", "0", "1"])
    a = ap.parse_args()

    files = ([a.path] if a.path.endswith(".nwb")
             else sorted(glob.glob(os.path.join(a.path, "**", "*.nwb"), recursive=True)))
    if not files:
        raise SystemExit(f"no .nwb files under {a.path}")

    print(f"converting {len(files)} file(s), events={a.event_col}, rewarded={a.rewarded}")
    ok = 0
    for f in files:
        try:
            csv_path, n_ev, fs = convert(f, a.outdir, a.event_col, a.rewarded)
            print(f"  {os.path.basename(csv_path):45s} {n_ev:4d} events  {fs:.1f} Hz")
            ok += 1
        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {e}")
    print(f"\n{ok}/{len(files)} converted -> {a.outdir}/")


if __name__ == "__main__":
    main()
