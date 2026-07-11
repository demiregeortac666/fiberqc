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


def test_opposite_signs_are_not_robust(clean_recording):
    """Regression: half the pipelines finding a significant INCREASE and half a
    significant DECREASE is the most choice-sensitive outcome there is, but the
    old verdict only counted p-values and called it ROBUST. Found on real
    DANDI:001340 data, where the bleaching method flipped the sign of the
    dopamine reward response (double_exp t=+13.7, highpass t=-5.6)."""
    keys = ["bleaching"]
    rows = [
        {"bleaching": "double_exp", "effect": 0.005, "t": +13.7, "p": 1e-33, "sig": True},
        {"bleaching": "double_exp", "effect": 0.004, "t": +12.3, "p": 1e-28, "sig": True},
        {"bleaching": "highpass",   "effect": -0.004, "t": -5.6, "p": 4e-08, "sig": True},
        {"bleaching": "highpass",   "effect": -0.003, "t": -4.2, "p": 4e-05, "sig": True},
    ]
    r = fqc.MultiverseResult(rows, keys, clean_recording)
    assert r.n_significant == 4          # every pipeline is "significant"...
    assert not r.signs_agree             # ...but they disagree on the direction
    assert r.verdict == "FLIP"           # so it must NOT be called ROBUST


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


# --------------------------------------------------------------- sampling gaps
def _gappy_csv(tmp_path):
    """A recording interrupted by a 140 s gap, like the real DANDI:001340 data."""
    import pandas as pd
    fs = 30.5
    t1 = np.arange(0, 400, 1 / fs)                    # 400 s block
    t2 = np.arange(540, 5000, 1 / fs)                 # resumes 140 s later
    t = np.concatenate([t1, t2])
    rng = np.random.default_rng(11)
    sig = 1 + 0.05 * rng.standard_normal(len(t))
    ctl = 1 + 0.05 * rng.standard_normal(len(t))
    p = tmp_path / "gappy.csv"
    pd.DataFrame({"time": t, "signal": sig, "control": ctl}).to_csv(p, index=False)
    return str(p), t


def test_gap_is_detected_and_longest_segment_kept(tmp_path):
    """Regression: load() used to accept a gappy recording silently and hand it to
    filters that assume uniform sampling. Found on real DANDI:001340 data, where a
    140 s LED-off gap rang through a 0.001 Hz high-pass and inverted the response."""
    path, t = _gappy_csv(tmp_path)
    with pytest.warns(UserWarning, match="not uniform"):
        rec = fqc.load(path)
    # the 4460 s segment must be kept, not the 400 s one, and not the whole array
    assert rec.time_s[-1] - rec.time_s[0] > 4000
    assert len(rec.signal) < len(t)
    assert np.max(np.diff(rec.time_s)) < 1.0          # no gap left inside


def test_gap_can_raise(tmp_path):
    path, _ = _gappy_csv(tmp_path)
    with pytest.raises(ValueError, match="not uniform"):
        fqc.load(path, on_gaps="raise")


def test_gap_events_outside_segment_are_dropped(tmp_path):
    path, _ = _gappy_csv(tmp_path)
    events = np.array([100.0, 200.0, 1000.0, 2000.0, 3000.0])   # 2 in the short block
    with pytest.warns(UserWarning):
        rec = fqc.load(path, events=events)
    assert len(rec.events) == 3                        # only the ones in the long segment
    assert rec.events.min() >= 540


# ------------------------------------------------------- effect size vs t-stat
def test_effect_size_is_reported_not_just_t(clean_recording):
    """Regression: the tool used to expose only the t-statistic. t mixes magnitude
    with variability and grows with sqrt(n), so a trivially small effect measured
    over many events looks as convincing as a large one (Chen et al., 2017).
    Every row must carry the raw effect, Cohen's d, and the event count."""
    r = fqc.multiverse(clean_recording, axes=_OLS_AXES)
    for row in r.rows:
        assert "effect" in row and "d" in row and "n_events" in row
    s = r.summary()
    for k in ("effect_min", "effect_max", "d_min", "d_max"):
        assert k in s
    assert "d=" in repr(r)          # magnitude is visible without opening the table


def test_t_is_d_times_sqrt_n(clean_recording):
    """t = d * sqrt(n). This identity is *why* the t-statistic is not an effect
    size: it is the effect size multiplied by the square root of the event count.
    Two recordings with the same d but 10x the events differ 3.2x in t while the
    finding is identical (Chen et al., 2017). Reporting d alongside t is what
    keeps that visible."""
    r = fqc.multiverse(clean_recording, axes=_OLS_AXES)
    for row in r.rows:
        expected_t = row["d"] * np.sqrt(row["n_events"])
        assert np.isclose(row["t"], expected_t, rtol=1e-6), (
            f"t={row['t']:.4f} but d*sqrt(n)={expected_t:.4f}")
