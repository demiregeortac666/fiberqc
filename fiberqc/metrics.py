"""
fiberqc.metrics
---------------
Pluggable downstream metrics. Each returns one value per event, so the
multiverse can t-test it against zero ("is there a reliable effect?") and check
whether that survives across pipelines.

Built-in metrics (name -> function):
    "mean"  mean(response window) - mean(baseline window)
    "peak"  max(response window)  - mean(baseline window)
    "auc"   area under the baseline-subtracted response window

You can also pass your own callable: metric(t_win, W) -> per-event array, where
W is the (n_events, n_samples) peri-event matrix and t_win the time axis.
Return values where 0 means "no effect" so the significance test is meaningful.
"""

import numpy as np

from .core import peri_event

try:                                    # np.trapz was renamed in NumPy 2.0
    from numpy import trapezoid as _trapz
except ImportError:
    from numpy import trapz as _trapz


def _split(t_win, W, baseline, response):
    b = W[:, (t_win >= baseline[0]) & (t_win < baseline[1])]
    r = W[:, (t_win >= response[0]) & (t_win < response[1])]
    return b, r


def mean_amplitude(t_win, W, baseline, response):
    b, r = _split(t_win, W, baseline, response)
    return r.mean(axis=1) - b.mean(axis=1)


def peak_amplitude(t_win, W, baseline, response):
    b, r = _split(t_win, W, baseline, response)
    return r.max(axis=1) - b.mean(axis=1)


def auc(t_win, W, baseline, response):
    b, r = _split(t_win, W, baseline, response)
    dt = float(np.median(np.diff(t_win)))
    return _trapz(r - b.mean(axis=1, keepdims=True), dx=dt, axis=1)


METRICS = {"mean": mean_amplitude, "peak": peak_amplitude, "auc": auc}


def compute(metric, trace, time_s, events, baseline, response, pre, post):
    """Compute per-event metric values for one processed trace."""
    t_win, W = peri_event(trace, time_s, events, pre=pre, post=post)
    if callable(metric):
        return np.asarray(metric(t_win, W), float)
    fn = METRICS.get(metric)
    if fn is None:
        raise ValueError(f"unknown metric {metric!r}; choose {list(METRICS)} or pass a callable")
    return fn(t_win, W, baseline, response)
