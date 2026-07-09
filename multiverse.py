"""
multiverse.py
-------------
The core of FP-Robust. Run the downstream analysis (peri-event reward response)
through every reasonable preprocessing pipeline, test whether the effect holds in
each, and draw a specification curve.

Metric per variant: per-event response amplitude = mean(dF/F in [0,2]s) minus
mean(baseline in [-1,0]s), then a one-sample t-test across events. The
t-statistic is unit-free, so it is comparable across dF/F and z-score variants;
significance is p < 0.05.

Outputs: spec_curve.png  and  multiverse_results.csv
Takes ~1-2 min (curve fits + robust regressions on the full recording).
"""

import os
import sys
import csv
import itertools
import numpy as np
from scipy.stats import ttest_1samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import preprocess, peri_event

HERE = os.path.dirname(os.path.abspath(__file__))
AKAM = os.path.join(HERE, "..", "photometry_preprocessing")
sys.path.insert(0, AKAM)
from data_import import import_ppd
DATA_DIR = os.path.join(AKAM, "data")
PPD = "m53_NAc_L-2019-11-24-093939.ppd"

# The multiverse: 2 x 2 x 2 x 2 = 16 defensible pipelines.
AXES = {
    "low_pass":      [10.0, 2.0],
    "bleaching":     ["double_exp", "highpass"],
    "motion":        ["OLS", "robust"],
    "normalization": ["dFF", "zscore"],
}

BASELINE_WIN = (-1.0, 0.0)
RESPONSE_WIN = (0.0, 2.0)


def event_amplitudes(trace, time_s, events):
    """Per-event response amplitude: mean(response window) - mean(baseline window)."""
    t_win, W = peri_event(trace, time_s, events)
    base = W[:, (t_win >= BASELINE_WIN[0]) & (t_win < BASELINE_WIN[1])].mean(axis=1)
    resp = W[:, (t_win >= RESPONSE_WIN[0]) & (t_win < RESPONSE_WIN[1])].mean(axis=1)
    return resp - base


def run():
    data = import_ppd(os.path.join(DATA_DIR, PPD))
    signal, control = data["analog_1"], data["analog_2"]
    time_s, fs = data["time"] / 1000, data["sampling_rate"]
    events = np.load(os.path.join(DATA_DIR, "reward_cue_times.npy"))
    if events.max() > time_s[-1] * 1.5:
        events = events / 1000

    keys = list(AXES)
    combos = list(itertools.product(*AXES.values()))
    rows = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        trace = preprocess(signal, control, time_s, fs, **params)
        amp = event_amplitudes(trace, time_s, events)
        t, p = ttest_1samp(amp, 0.0)
        rows.append({**params,
                     "effect": float(amp.mean()),
                     "t": float(t), "p": float(p), "sig": bool(p < 0.05)})
        print(f"  [{i:>2}/16] {combo}  t={t:6.2f}  p={p:.1e}  "
              f"{'SIG' if p < 0.05 else 'ns'}")

    rows.sort(key=lambda r: r["t"])
    return keys, rows


def report(keys, rows):
    n_sig = sum(r["sig"] for r in rows)
    ts = [r["t"] for r in rows]
    print("\n---------------- MULTIVERSE ----------------")
    print(f"{n_sig}/{len(rows)} specifications significant (p<0.05)")
    print(f"effect (t) range: {min(ts):.2f} .. {max(ts):.2f}  "
          f"(fold: {max(ts)/min(ts):.1f}x)")
    if n_sig == len(rows):
        print("VERDICT: result is ROBUST -- reward response survives every pipeline.")
    elif n_sig == 0:
        print("VERDICT: result vanishes under every pipeline.")
    else:
        print(f"VERDICT: result FLIPS -- significant in {n_sig}, not in {len(rows)-n_sig}.")

    with open(os.path.join(HERE, "multiverse_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["effect", "t", "p", "sig"])
        w.writeheader()
        w.writerows(rows)


def plot(keys, rows):
    n = len(rows)
    x = np.arange(n)
    ts = np.array([r["t"] for r in rows])
    sig = np.array([r["sig"] for r in rows])
    tcrit = 1.98  # ~two-sided .05, df~136

    options = [(k, v) for k in keys for v in AXES[k]]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 3]})

    # top: effect (t) per specification, colored by significance
    ax1.axhline(tcrit, color="gray", ls="--", lw=1, label=f"p=.05 (t={tcrit})")
    ax1.scatter(x[sig], ts[sig], c="#1a7f37", s=40, label="significant", zorder=3)
    ax1.scatter(x[~sig], ts[~sig], c="#cf222e", s=40, label="not significant", zorder=3)
    ax1.set_ylabel("effect  (t-statistic)")
    ax1.set_title("Specification curve — reward response across 16 pipelines")
    ax1.legend(loc="upper left", fontsize=9)

    # bottom: which choice each specification used
    for row_i, (k, v) in enumerate(options):
        ys = np.full(n, row_i, float)
        used = np.array([rows[c][k] == v for c in range(n)])
        ax2.scatter(x[used], ys[used], c="#24292f", s=22)
        ax2.scatter(x[~used], ys[~used], c="#d0d7de", s=6)
    ax2.set_yticks(range(len(options)))
    ax2.set_yticklabels([f"{k} = {v}" for k, v in options], fontsize=9)
    ax2.set_xlabel("specification (sorted by effect size)")
    ax2.invert_yaxis()

    plt.tight_layout()
    out = os.path.join(HERE, "spec_curve.png")
    plt.savefig(out, dpi=120)
    print(f"saved -> {out}")


if __name__ == "__main__":
    keys, rows = run()
    report(keys, rows)
    plot(keys, rows)
