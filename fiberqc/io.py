"""
fiberqc.io
----------
Loading. `load()` returns a Recording from a pyPhotometry .ppd file or any
two-channel CSV (signal + control). Not locked to one vendor format.
"""

import os
import warnings
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


def _handle_gaps(signal, control, time_s, events, name, max_gap=5.0, on_gaps="longest"):
    """Recordings are often interrupted (the LED is switched off between blocks).
    Every filter in preprocess() assumes uniform sampling, so a gap is not a
    harmless missing stretch: it is a step discontinuity that rings through the
    whole trace. A 0.001 Hz high-pass has a ~1000 s time constant, so a single
    140 s gap can invert the sign of the measured response.

    So gaps are never ignored. `on_gaps` chooses what to do:
      "longest" (default) keep the longest continuous segment, warn, and drop
                the events that fall outside it
      "raise"   refuse to load
      "ignore"  proceed anyway (you are on your own)
    """
    t = np.asarray(time_s, float)
    if len(t) < 3 or on_gaps == "ignore":
        return signal, control, t, events

    dt = np.diff(t)
    step = float(np.median(dt))
    gap_idx = np.where(dt > max_gap * step)[0]
    if len(gap_idx) == 0:
        return signal, control, t, events

    gaps = dt[gap_idx]
    msg = (f"{name}: sampling is not uniform — {len(gap_idx)} gap(s), "
           f"largest {gaps.max():.1f} s (median step {step*1000:.1f} ms). "
           "Filtering assumes uniform sampling, so gaps corrupt the result.")
    if on_gaps == "raise":
        raise ValueError(msg + " Pass on_gaps='longest' to analyse the longest "
                               "continuous segment instead.")

    edges = np.concatenate(([0], gap_idx + 1, [len(t)]))
    segments = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)
                if edges[i + 1] - edges[i] > 1]
    a, b = max(segments, key=lambda s: t[s[1] - 1] - t[s[0]])
    kept = t[b - 1] - t[a]
    total = t[-1] - t[0]

    if events is not None:
        events = np.asarray(events, float)
        n_before = len(events)
        events = events[(events >= t[a]) & (events <= t[b - 1])]
        dropped = n_before - len(events)
    else:
        dropped = 0

    warnings.warn(msg + f" Using the longest continuous segment: {kept:.0f} s of "
                        f"{total:.0f} s" +
                  (f", dropping {dropped} event(s) outside it." if dropped else "."),
                  stacklevel=2)
    return signal[a:b], control[a:b], t[a:b], events


def load(path, events=None, *, signal_col=None, control_col=None,
         time_col=None, fs=None, ppd_dir=None, name=None,
         max_gap=5.0, on_gaps="longest"):
    """Load a fiber photometry recording into a :class:`Recording`.

    Parameters
    ----------
    path : str
        Path to the recording. A pyPhotometry ``.ppd`` file, or any two-column
        CSV/TSV with a signal and a control channel.
    events : str or array-like, optional
        Event times in seconds. Either a path (``.npy`` or ``.csv``) or an array.
        If omitted, the recording has no events and only
        :func:`fiberqc.motion_robustness` can be run on it.
    signal_col, control_col, time_col : str, optional
        Column names for CSV input. If omitted, they are auto-detected from
        common names (e.g. ``470``/``dLight``/``signal`` and ``405``/``iso``/``control``).
    fs : float, optional
        Sampling rate in Hz. Required for CSV input that has no time column.
    ppd_dir : str, optional
        Path to a clone of Akam's ``photometry_preprocessing`` repository, whose
        ``.ppd`` reader is used for pyPhotometry files.
    name : str, optional
        Display name for the recording. Defaults to the file name.
    max_gap : float, optional
        A jump between samples longer than ``max_gap`` times the median step is
        treated as a gap in the recording. Default 5.
    on_gaps : {"longest", "raise", "ignore"}, optional
        What to do when the recording is not uniformly sampled. ``"longest"``
        (default) keeps the longest continuous segment and warns; ``"raise"``
        refuses to load; ``"ignore"`` proceeds anyway. Gaps matter because every
        filter assumes uniform sampling — a single long gap is a step
        discontinuity that can invert the measured response.

    Returns
    -------
    Recording
        The loaded recording, ready for :func:`fiberqc.multiverse`.

    Examples
    --------
    >>> import fiberqc as fqc
    >>> rec = fqc.load("m53.ppd", events="m53_events.npy", ppd_dir="../photometry_preprocessing")
    >>> rec
    Recording('m53.ppd', 78000 samples, 130 Hz, 137 events)
    """
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

    signal, control, time_s, ev = _handle_gaps(signal, control, time_s, ev,
                                               name or os.path.basename(path),
                                               max_gap=max_gap, on_gaps=on_gaps)

    return Recording(signal=signal, control=control, time_s=time_s, fs=fs,
                     events=ev, name=name or os.path.basename(path))
