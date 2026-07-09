"""
data_io.py
----------
Generic loader so FP-Robust is not locked to one vendor format.

Supports:
  * pyPhotometry .ppd   (via Akam's import_ppd, referenced from the sibling clone)
  * CSV / TSV / TXT      (any two-channel table: signal + control, plus time or fs)

CSV convention: columns are auto-detected by name (signal/gcamp/dlight/470...,
control/isosbestic/405/tdtomato...). If detection fails, pass the column names
explicitly. Time comes from a time column (auto-detected) or from `fs=`.
"""

import os
import sys
import numpy as np


def _load_ppd(path):
    here = os.path.dirname(os.path.abspath(__file__))
    akam = os.path.join(here, "..", "photometry_preprocessing")
    sys.path.insert(0, akam)
    from data_import import import_ppd
    d = import_ppd(path)
    return d["analog_1"], d["analog_2"], d["time"] / 1000, d["sampling_rate"]


_SIGNAL_HINTS  = ["signal", "gcamp", "dlight", "grab", "470", "465", "490", "dff", "green"]
_CONTROL_HINTS = ["control", "isosbestic", "iso", "405", "410", "tdtomato", "tdtom", "red"]
_TIME_HINTS    = ["time", "timestamp", "seconds", "sec"]


def _pick(columns, hints):
    for h in hints:
        for c in columns:
            if h in c.lower():
                return c
    return None


def _load_csv(path, signal_col, control_col, time_col, fs):
    import pandas as pd
    df = pd.read_csv(path, sep=None, engine="python")
    cols = list(df.columns)

    signal_col  = signal_col  or _pick(cols, _SIGNAL_HINTS)
    control_col = control_col or _pick(cols, _CONTROL_HINTS)
    if signal_col is None or control_col is None:
        raise ValueError(
            f"Could not auto-detect signal/control columns in {cols}. "
            "Pass signal_col=/control_col= explicitly.")

    signal  = df[signal_col].to_numpy(float)
    control = df[control_col].to_numpy(float)

    time_col = time_col or _pick(cols, _TIME_HINTS)
    if time_col:
        t = df[time_col].to_numpy(float)
        dt = np.median(np.diff(t))
        if dt > 1:                       # looks like milliseconds -> seconds
            t, dt = t / 1000, dt / 1000
        time_s, fs = t, 1 / dt
    elif fs:
        time_s = np.arange(len(signal)) / fs
    else:
        raise ValueError("No time column found; pass fs= (sampling rate in Hz).")
    return signal, control, time_s, fs


def load_events(path):
    if path.endswith(".npy"):
        return np.load(path)
    return np.loadtxt(path, delimiter=",").ravel()


def load_recording(path, events_path=None, *, signal_col=None, control_col=None,
                   time_col=None, fs=None):
    """Return dict(signal, control, time_s, fs, events)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ppd":
        signal, control, time_s, fs = _load_ppd(path)
    elif ext in (".csv", ".tsv", ".txt"):
        signal, control, time_s, fs = _load_csv(path, signal_col, control_col, time_col, fs)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    events = load_events(events_path) if events_path else None
    return {"signal": signal, "control": control, "time_s": time_s, "fs": fs, "events": events}
