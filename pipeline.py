"""
pipeline.py
-----------
Parametrized fiber photometry preprocessing, refactored from Thomas Akam's
baseline notebook (photometry_preprocessing, GPL3 -- referenced, not vendored).

The whole point of FP-Robust is that each preprocessing step is a *choice*.
So instead of one fixed pipeline, `preprocess()` exposes each choice as an
argument. Calling it with the defaults reproduces Akam's baseline exactly;
flipping an argument gives you one point in the multiverse.

Axes exposed:
    low_pass       : low-pass cutoff in Hz (denoising)         [baseline 10]
    median_filter  : median filter window in samples, or None  [baseline None]
    bleaching      : 'double_exp' | 'highpass'                 [baseline 'double_exp']
    motion         : 'OLS' | 'robust'                          [baseline 'OLS']
    normalization  : 'dFF' | 'zscore'                          [baseline 'dFF']
"""

import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import statsmodels.api as sm


# ---------------------------------------------------------------- bleaching
def _double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    tau_fast = tau_slow * tau_multiplier
    return const + amp_slow * np.exp(-t / tau_slow) + amp_fast * np.exp(-t / tau_fast)


def _fit_double_exp(t, y):
    """Fit Akam's double-exponential bleaching curve; returns the fitted trace."""
    max_sig = np.max(y)
    p0 = [max_sig / 2, max_sig / 4, max_sig / 4, 3600, 0.1]
    bounds = ([0, 0, 0, 600, 0], [max_sig, max_sig, max_sig, 36000, 1])
    params, _ = curve_fit(_double_exponential, t, y, p0=p0, bounds=bounds, maxfev=2000)
    return _double_exponential(t, *params)


# ---------------------------------------------------------------- pipeline
def preprocess(signal, control, time_s, fs, *,
               low_pass=10.0, median_filter=None,
               bleaching="double_exp", motion="OLS", normalization="dFF",
               return_info=False):
    """Run one pipeline variant. Defaults == Akam baseline.

    Returns the processed trace (same length as input), or (trace, info) if
    return_info=True. `info` carries diagnostics (motion slope/R^2, F0) useful
    for verifying against the notebook.
    """
    sig = np.asarray(signal, float)
    ctl = np.asarray(control, float)
    t = np.asarray(time_s, float)

    # 1. Denoise: optional median filter, then zero-phase low-pass.
    if median_filter:
        k = int(median_filter)
        k += 1 - (k % 2)                    # force odd window
        sig, ctl = medfilt(sig, k), medfilt(ctl, k)
    if low_pass:
        b, a = butter(2, low_pass, btype="low", fs=fs)
        sig, ctl = filtfilt(b, a, sig), filtfilt(b, a, ctl)
    denoised = sig

    # F0 for dF/F is always the double-exp bleaching baseline of the signal,
    # so dF/F stays well-defined regardless of the detrend method chosen below.
    F0 = _fit_double_exp(t, denoised)

    # 2. Bleaching correction (detrend both channels).
    if bleaching == "double_exp":
        sig_d = denoised - F0
        ctl_d = ctl - _fit_double_exp(t, ctl)
    elif bleaching == "highpass":
        b, a = butter(2, 0.001, btype="high", fs=fs)
        sig_d = filtfilt(b, a, denoised, padtype="even")
        ctl_d = filtfilt(b, a, ctl, padtype="even")
    else:
        raise ValueError(f"bleaching={bleaching!r}")

    # 3. Motion correction: regress control -> signal, subtract the estimate.
    if motion == "OLS":
        slope, intercept, r, p, se = linregress(x=ctl_d, y=sig_d)
        r2 = r ** 2
    elif motion == "robust":
        res = sm.RLM(sig_d, sm.add_constant(ctl_d),
                     M=sm.robust.norms.HuberT()).fit()
        intercept, slope = res.params[0], res.params[1]
        r2 = np.nan                          # RLM has no direct R^2
    else:
        raise ValueError(f"motion={motion!r}")
    corrected = sig_d - (intercept + slope * ctl_d)

    # 4. Normalization.
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
def peri_event(trace, time_s, event_times, pre=1.0, post=3.0):
    """Align `trace` to each event. Returns (t_win, W) where W is
    (n_events_kept, n_samples)."""
    t = np.asarray(time_s, float)
    dt = np.median(np.diff(t))
    n_pre, n_post = int(round(pre / dt)), int(round(post / dt))
    idx = np.searchsorted(t, np.asarray(event_times, float))
    W = [trace[i - n_pre:i + n_post] for i in idx
         if i - n_pre >= 0 and i + n_post <= len(trace)]
    W = np.array(W)
    t_win = np.arange(-n_pre, n_post) * dt
    return t_win, W


def peri_event_peak(trace, time_s, event_times,
                    pre=1.0, post=3.0, baseline=(-1.0, 0.0), response=(0.0, 2.0)):
    """Baseline-subtracted peak of the event-averaged response.
    Returns (peak, t_win, mean_response, W)."""
    t_win, W = peri_event(trace, time_s, event_times, pre, post)
    mean_resp = W.mean(axis=0)
    base = mean_resp[(t_win >= baseline[0]) & (t_win < baseline[1])].mean()
    peak = (mean_resp[(t_win >= response[0]) & (t_win < response[1])] - base).max()
    return peak, t_win, mean_resp, W
