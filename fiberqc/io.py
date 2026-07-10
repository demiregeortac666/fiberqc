"""
fiberqc.io
----------
Loading. `load()` returns a Recording from a pyPhotometry .ppd file or any
two-channel CSV (signal + control). Not locked to one vendor format.
"""

import os
import sys

import numpy as np

from .core import Recording

_SIGNAL_HINTS  = ["signal", "gcamp", "dlight", "grab", "470", "465", "490", "dff", "green"]
_CONTROL_HINTS = ["control", "isosbestic", "iso", "405", "410", "tdtomato", "tdtom", "red"]
_TIME_HINTS    = ["time", "timestamp", "seconds", "sec"]


def _pick(columns, hints):
    for h in hints:
        for c in columns:
            if h in c.lower():
                return c
    return None


def _load_ppd(path, ppd_dir=None):
    # pyPhotometry loader lives in Akam's repo; allow pointing at it.
    if ppd_dir:
        sys.path.insert(0, ppd_dir)
    from data_import import import_ppd
    d = import_ppd(path)
    return d["analog_1"], d["analog_2"], d["time"] / 1000, d["sampling_rate"]


def _load_csv(path, signal_col, control_col, time_col, fs):
    import pandas as pd
    df = pd.read_csv(path, sep=None, engine="python")
    cols = list(df.columns)
    signal_col  = signal_col  or _pick(cols, _SIGNAL_HINTS)
    control_col = control_col or _pick(cols, _CONTROL_HINTS)
    if signal_col is None or control_col is None:
        raise ValueError(f"Could not auto-detect signal/control columns in {cols}. "
                         "Pass signal_col=/control_col=.")
    signal  = df[signal_col].to_numpy(float)
    control = df[control_col].to_numpy(float)
    time_col = time_col or _pick(cols, _TIME_HINTS)
    if time_col:
        t = df[time_col].to_numpy(float)
        dt = np.median(np.diff(t))
        if dt > 1:
            t, dt = t / 1000, dt / 1000
        return signal, control, t, 1 / dt
    if fs:
        return signal, control, np.arange(len(signal)) / fs, fs
    raise ValueError("No time column found; pass fs=.")


def load_events(path):
    if path.endswith(".npy"):
        return np.load(path)
    return np.loadtxt(path, delimiter=",").ravel()


def load(path, events=None, *, signal_col=None, control_col=None,
         time_col=None, fs=None, ppd_dir=None, name=None):
    """Load a recording. `events` may be a path (.npy/.csv) or an array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ppd":
        signal, control, time_s, fs = _load_ppd(path, ppd_dir)
    elif ext in (".csv", ".tsv", ".txt"):
        signal, control, time_s, fs = _load_csv(path, signal_col, control_col, time_col, fs)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    ev = None
    if events is not None:
        ev = load_events(events) if isinstance(events, str) else np.asarray(events, float)
        if len(time_s) and ev.max() > time_s[-1] * 1.5:   # ms -> s guard
            ev = ev / 1000

    return Recording(signal=signal, control=control, time_s=time_s, fs=fs,
                     events=ev, name=name or os.path.basename(path))
