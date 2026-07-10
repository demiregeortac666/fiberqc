"""
fiberqc.core
------------
Recording container, the parametrized preprocessing pipeline, and the
peri-event metric. This is the scientific heart; the numbers here match the
original Akam baseline when called with defaults.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit

# Default multiverse axes (the choices the reference primer discusses).
DEFAULT_AXES = {
    "low_pass":      [10.0, 2.0],
    "bleaching":     ["double_exp", "highpass"],
    "motion":        ["OLS", "robust"],
    "normalization": ["dFF", "zscore"],
}

BASELINE_WIN = (-1.0, 0.0)
RESPONSE_WIN = (0.0, 2.0)


@dataclass
class Recording:
    """A two-channel fiber photometry recording."""
    signal: np.ndarray
    control: np.ndarray
    time_s: np.ndarray
    fs: float
    events: np.ndarray = None
    name: str = "recording"

    def __repr__(self):
        ev = "no events" if self.events is None else f"{len(self.events)} events"
        return (f"Recording({self.name!r}, {len(self.signal)} samples, "
                f"{self.fs:.0f} Hz, {ev})")


# ---------------------------------------------------------------- bleaching
def _double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    tau_fast = tau_slow * tau_multiplier
    return const + amp_slow * np.exp(-t / tau_slow) + amp_fast * np.exp(-t / tau_fast)


def _fit_double_exp(t, y):
    """Fit a double-exponential bleaching curve. If the fit fails to converge
    (can happen on short or heavily-filtered traces), fall back to a smooth
    low-order polynomial trend so a single hard pipeline never crashes the run."""
    max_sig = np.max(y)
    p0 = [max_sig / 2, max_sig / 4, max_sig / 4, 3600, 0.1]
    bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
    try:
        params, _ = curve_fit(_double_exponential, t, y, p0=p0,
                              bounds=bounds, maxfev=5000)
        return _double_exponential(t, *params)
    except (RuntimeError, ValueError):
        # least-squares polynomial trend as a robust fallback baseline
        coeffs = np.polyfit(t, y, 2)
        return np.polyval(coeffs, t)


# ---------------------------------------------------------------- pipeline
def preprocess(signal, control, time_s, fs, *,
               low_pass=10.0, median_filter=None,
               bleaching="double_exp", motion="OLS", normalization="dFF",
               return_info=False):
    """Run one pipeline variant. Defaults reproduce the Akam baseline."""
    sig = np.asarray(signal, float)
    ctl = np.asarray(control, float)
    t = np.asarray(time_s, float)

    if median_filter:
        k = int(median_filter)
        k += 1 - (k % 2)
        sig, ctl = medfilt(sig, k), medfilt(ctl, k)
    if low_pass:
        b, a = butter(2, low_pass, btype="low", fs=fs)
        sig, ctl = filtfilt(b, a, sig), filtfilt(b, a, ctl)
    denoised = sig

    F0 = _fit_double_exp(t, denoised)

    if bleaching == "double_exp":
        sig_d = denoised - F0
        ctl_d = ctl - _fit_double_exp(t, ctl)
    elif bleaching == "highpass":
        b, a = butter(2, 0.001, btype="high", fs=fs)
        sig_d = filtfilt(b, a, denoised, padtype="even")
        ctl_d = filtfilt(b, a, ctl, padtype="even")
    else:
        raise ValueError(f"bleaching={bleaching!r}")

    if motion == "OLS":
        slope, intercept, r, p, se = linregress(x=ctl_d, y=sig_d)
        r2 = r ** 2
    elif motion == "robust":
        import statsmodels.api as sm          # lazy: only needed for robust
        res = sm.RLM(sig_d, sm.add_constant(ctl_d),
                     M=sm.robust.norms.HuberT()).fit()
        intercept, slope = res.params[0], res.params[1]
        r2 = np.nan
    else:
        raise ValueError(f"motion={motion!r}")
    corrected = sig_d - (intercept + slope * ctl_d)

    if normalization == "dFF":
        out = 100 * corrected / F0
    elif normalization == "zscore":
        out = (corrected - corrected.mean()) / corrected.std()
    else:
        raise ValueError(f"normalization={normalization!r}")

    if return_info:
        return out, {"slope": slope, "r2": r2, "F0": F0}
    return out


# ---------------------------------------------------------------- metric
def peri_event(trace, time_s, events, pre=1.0, post=3.0):
    t = np.asarray(time_s, float)
    dt = np.median(np.diff(t))
    n_pre, n_post = int(round(pre / dt)), int(round(post / dt))
    idx = np.searchsorted(t, np.asarray(events, float))
    W = [trace[i - n_pre:i + n_post] for i in idx
         if i - n_pre >= 0 and i + n_post <= len(trace)]
    return np.arange(-n_pre, n_post) * dt, np.array(W)


def peri_event_amplitudes(trace, time_s, events,
                          baseline=BASELINE_WIN, response=RESPONSE_WIN):
    """Per-event amplitude: mean(response window) - mean(baseline window)."""
    t_win, W = peri_event(trace, time_s, events)
    base = W[:, (t_win >= baseline[0]) & (t_win < baseline[1])].mean(axis=1)
    resp = W[:, (t_win >= response[0]) & (t_win < response[1])].mean(axis=1)
    return resp - base
