"""
Test suite for fiberqc.

Uses deterministic synthetic recordings (fixed seeds) so results are
reproducible. The Claude interpretation layer (report/ask) is intentionally not
tested here: it requires network access and an API key, and "did the model write
good prose" is not a unit-testable property. Everything numerical is covered.
"""

import importlib.util

import numpy as np
import pytest

import fiberqc as fqc
from fiberqc.metrics import mean_amplitude, peak_amplitude, auc

_HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None
_OLS_AXES = {
    "low_pass": [10.0, 2.0],
    "bleaching": ["double_exp", "highpass"],
    "motion": ["OLS"],
    "normalization": ["dFF", "zscore"],
}


# --------------------------------------------------------------- fixtures
def _synth(seed, with_response):
    """A deterministic two-channel recording; optionally with an event response."""
    rng = np.random.default_rng(seed)
    fs, dur = 100.0, 300.0
    n = int(dur * fs)
    t = np.arange(n) / fs
    bleach = 1 + 2 * np.exp(-t / 200)
    control = bleach + 0.02 * rng.standard_normal(n)
    signal = bleach + 0.02 * rng.standard_normal(n)
    events = np.arange(10, dur - 10, 5.0)
    if with_response:
        for e in events:
            i = int(e * fs)
            tt = np.arange(0, int(2 * fs)) / fs
            signal[i:i + len(tt)] += 0.5 * np.exp(-tt / 0.4)
    return fqc.Recording(signal=signal, control=control, time_s=t, fs=fs,
                         events=events, name="synth")


@pytest.fixture
def clean_recording():
    return _synth(seed=42, with_response=True)


@pytest.fixture
def noise_recording():
    return _synth(seed=7, with_response=False)


# --------------------------------------------------------------- metrics
def _known_matrix():
    t_win = np.linspace(-1, 2, 301)
    W = np.zeros((5, 301))
    W[:, t_win >= 0] = 2.0          # baseline 0, response constant 2.0
    return t_win, W


def test_metric_mean_is_exact():
    t_win, W = _known_matrix()
    assert np.allclose(mean_amplitude(t_win, W, (-1, 0), (0, 2)), 2.0)


def test_metric_peak_is_exact():
    t_win, W = _known_matrix()
    assert np.allclose(peak_amplitude(t_win, W, (-1, 0), (0, 2)), 2.0)


def test_metric_auc_matches_area():
    t_win, W = _known_matrix()
    # area of a height-2 block over a 2 s window is ~4 (trapezoid, minus one step)
    assert np.allclose(auc(t_win, W, (-1, 0), (0, 2)), 4.0, atol=0.05)


# --------------------------------------------------------------- verdict
def test_robust_on_clean_signal(clean_recording):
    r = fqc.multiverse(clean_recording, axes=_OLS_AXES)
    assert r.verdict == "ROBUST"
    assert r.n_significant == r.n


def test_null_on_pure_noise(noise_recording):
    r = fqc.multiverse(noise_recording, axes=_OLS_AXES)
    assert r.verdict == "NULL"
    assert r.n_significant == 0


def test_verdict_logic_from_rows(clean_recording):
    """verdict is a pure function of how many pipelines were significant."""
    keys = ["low_pass"]
    def rows(sigs):
        return [{"low_pass": i, "effect": 1.0, "t": 5.0, "p": 0.01, "sig": s}
                for i, s in enumerate(sigs)]
    all_sig = fqc.MultiverseResult(rows([True, True, True]), keys, clean_recording)
    none_sig = fqc.MultiverseResult(rows([False, False, False]), keys, clean_recording)
    mixed = fqc.MultiverseResult(rows([True, False, True]), keys, clean_recording)
    assert all_sig.verdict == "ROBUST"
    assert none_sig.verdict == "NULL"
    assert mixed.verdict == "FLIP"


# --------------------------------------------------------------- flexibility
def test_all_builtin_metrics_run(clean_recording):
    for m in ("mean", "peak", "auc"):
        r = fqc.multiverse(clean_recording, axes=_OLS_AXES, metric=m)
        assert r.verdict == "ROBUST"
        assert r.metric == m


def test_custom_metric_callable(clean_recording):
    def frac_positive(t_win, W):
        base = W[:, (t_win >= -1) & (t_win < 0)].mean(axis=1, keepdims=True)
        resp = W[:, (t_win >= 0) & (t_win < 2)]
        return (resp > base).mean(axis=1) - 0.5
    r = fqc.multiverse(clean_recording, axes=_OLS_AXES, metric=frac_positive)
    assert r.n == 8                      # 2 x 2 x 1 x 2
    assert callable(r.metric)


def test_custom_axes_pipeline_count(clean_recording):
    axes = {"low_pass": [1.0, 5.0, 10.0], "median_filter": [None, 5],
            "motion": ["OLS"], "normalization": ["dFF"]}
    r = fqc.multiverse(clean_recording, axes=axes)
    assert r.n == 6                      # 3 x 2 x 1 x 1


def test_configurable_windows(clean_recording):
    r = fqc.multiverse(clean_recording, axes=_OLS_AXES,
                       baseline=(-2, 0), response=(0, 1))
    assert r.baseline == (-2, 0)
    assert r.response == (0, 1)


# --------------------------------------------------------------- io
def test_csv_autodetect(tmp_path):
    import pandas as pd
    fs = 100.0
    t = np.arange(1000) / fs
    df = pd.DataFrame({"time": t,
                       "dLight_470": np.sin(t) + 1,
                       "iso_405": np.cos(t) + 1})
    p = tmp_path / "rec.csv"
    df.to_csv(p, index=False)
    rec = fqc.load(str(p))
    assert len(rec.signal) == 1000
    assert abs(rec.fs - fs) < 1


def test_multiverse_requires_events(clean_recording):
    rec = fqc.Recording(signal=clean_recording.signal, control=clean_recording.control,
                        time_s=clean_recording.time_s, fs=clean_recording.fs,
                        events=None, name="noev")
    with pytest.raises(ValueError):
        fqc.multiverse(rec)


# --------------------------------------------------------------- preprocess
@pytest.mark.parametrize("bleaching", ["double_exp", "highpass"])
@pytest.mark.parametrize("normalization", ["dFF", "zscore"])
def test_preprocess_options_finite(clean_recording, bleaching, normalization):
    out = fqc.preprocess(clean_recording.signal, clean_recording.control,
                         clean_recording.time_s, clean_recording.fs,
                         bleaching=bleaching, normalization=normalization)
    assert len(out) == len(clean_recording.signal)
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------- result io
def test_to_df_has_expected_columns(clean_recording):
    r = fqc.multiverse(clean_recording, axes=_OLS_AXES)
    df = r.to_df()
    for col in ("t", "p", "sig", "effect"):
        assert col in df.columns
    assert len(df) == r.n


# --------------------------------------------------------------- robust (needs statsmodels)
@pytest.mark.skipif(not _HAS_STATSMODELS, reason="statsmodels not installed")
def test_robust_motion_runs(clean_recording):
    axes = {"low_pass": [10.0], "bleaching": ["double_exp"],
            "motion": ["OLS", "robust"], "normalization": ["dFF"]}
    r = fqc.multiverse(clean_recording, axes=axes)
    assert r.n == 2
    assert r.verdict in ("ROBUST", "FLIP", "NULL")
