"""
evidence.py
-----------
Auditable evidence for every pipeline in the multiverse -- so nothing is taken
on trust. For each of the 16 pipelines this writes, into ./variants/:

  * <name>_periavg.png    the peri-event average for that pipeline (visual proof)
  * <name>_amplitudes.csv the raw per-event amplitudes (the t-test's actual input)

Plus ./manifest.csv: one row per pipeline with its full parameters, t, p,
significance, and the paths to its evidence files -- the index of the bundle.

An auditor can open any amplitudes CSV, run their own one-sample t-test, and
reproduce the exact t/p reported. Numbers come from the data, not from us.
"""

import os
import csv
import sys
import itertools

import numpy as np
from scipy.stats import ttest_1samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import preprocess, peri_event
from data_io import load_recording

AXES = {
    "low_pass":      [10.0, 2.0],
    "bleaching":     ["double_exp", "highpass"],
    "motion":        ["OLS", "robust"],
    "normalization": ["dFF", "zscore"],
}
BASELINE_WIN = (-1.0, 0.0)
RESPONSE_WIN = (0.0, 2.0)
OUTDIR = "variants"


def name_of(params):
    return f"lp{params['low_pass']:g}_{params['bleaching']}_{params['motion']}_{params['normalization']}"


def per_event(trace, time_s, events):
    """Return (t_win, W, amplitudes) for one trace."""
    t_win, W = peri_event(trace, time_s, events)
    base = W[:, (t_win >= BASELINE_WIN[0]) & (t_win < BASELINE_WIN[1])].mean(axis=1)
    resp = W[:, (t_win >= RESPONSE_WIN[0]) & (t_win < RESPONSE_WIN[1])].mean(axis=1)
    return t_win, W, resp - base


def save_plot(path, t_win, W, params, t, p):
    mean = W.mean(axis=0)
    sem = W.std(axis=0) / np.sqrt(len(W))
    plt.figure(figsize=(6, 3.5))
    plt.axvline(0, color="k", lw=0.8, ls="--")
    plt.plot(t_win, mean, "g")
    plt.fill_between(t_win, mean - sem, mean + sem, color="g", alpha=0.2)
    plt.xlabel("Time from event (s)")
    plt.ylabel("signal (variant units)")
    plt.title(f"{name_of(params)}\nt={t:.2f}  p={p:.1e}", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    ap_data = sys.argv[1] if len(sys.argv) > 1 else \
        "../photometry_preprocessing/data/m53_NAc_L-2019-11-24-093939.ppd"
    ap_events = sys.argv[2] if len(sys.argv) > 2 else \
        "../photometry_preprocessing/data/reward_cue_times.npy"

    rec = load_recording(ap_data, ap_events)
    signal, control = rec["signal"], rec["control"]
    time_s, fs, events = rec["time_s"], rec["fs"], rec["events"]
    if events.max() > time_s[-1] * 1.5:
        events = events / 1000

    os.makedirs(OUTDIR, exist_ok=True)
    keys = list(AXES)
    manifest = []

    for combo in itertools.product(*AXES.values()):
        params = dict(zip(keys, combo))
        nm = name_of(params)
        trace = preprocess(signal, control, time_s, fs, **params)
        t_win, W, amp = per_event(trace, time_s, events)
        t, p = ttest_1samp(amp, 0.0)

        # 1) raw per-event amplitudes (the t-test's input)
        amp_path = os.path.join(OUTDIR, f"{nm}_amplitudes.csv")
        with open(amp_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event_index", "amplitude"])
            for i, a in enumerate(amp):
                w.writerow([i, a])

        # 2) per-pipeline peri-event plot
        png_path = os.path.join(OUTDIR, f"{nm}_periavg.png")
        save_plot(png_path, t_win, W, params, t, p)

        manifest.append({**params, "n_events": len(amp),
                         "t": round(float(t), 4), "p": float(p),
                         "sig": bool(p < 0.05),
                         "amplitudes_file": amp_path, "plot_file": png_path})
        print(f"  {nm:38s} t={t:6.2f}  p={p:.1e}  -> {os.path.basename(amp_path)}")

    # 3) manifest index
    with open("manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)

    n_sig = sum(r["sig"] for r in manifest)
    print(f"\n{len(manifest)} pipelines -> {n_sig} significant.")
    print(f"Evidence in ./{OUTDIR}/  ({2*len(manifest)} files) + manifest.csv")
    print("Audit check: open any *_amplitudes.csv, run a one-sample t-test, "
          "and you get that row's t/p.")


if __name__ == "__main__":
    main()
