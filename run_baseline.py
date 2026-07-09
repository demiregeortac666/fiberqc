"""
run_baseline.py
---------------
End-to-end smoke test of the baseline pipeline on Akam's NAc/dLight recording.

Loads the data via the sibling photometry_preprocessing clone (referenced, not
copied -- it's GPL3, we keep it out of this repo), runs preprocess() with the
default (= baseline) settings, prints diagnostics, and computes the downstream
metric (peri-event dF/F peak around reward cues). Saves an average-response plot.

Green check: the printed motion slope / R^2 should match notebook cell 22.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import preprocess, peri_event_peak

# --- locate Akam's clone (sibling dir) for the loader + data --------------
HERE = os.path.dirname(os.path.abspath(__file__))
AKAM = os.path.join(HERE, "..", "photometry_preprocessing")
sys.path.insert(0, AKAM)
from data_import import import_ppd            # Akam's GPL3 loader, used in place

DATA_DIR = os.path.join(AKAM, "data")
PPD = "m53_NAc_L-2019-11-24-093939.ppd"


def main():
    data = import_ppd(os.path.join(DATA_DIR, PPD))
    signal  = data["analog_1"]               # dLight
    control = data["analog_2"]               # TdTomato control
    time_s  = data["time"] / 1000            # ms -> s
    fs      = data["sampling_rate"]
    duration = time_s[-1]

    events = np.load(os.path.join(DATA_DIR, "reward_cue_times.npy"))
    # sanity: reward_cue_times should be in seconds; if they look like ms, fix.
    if events.max() > duration * 1.5:
        print("note: event times look like ms -> converting to s")
        events = events / 1000

    print(f"samples={len(signal)}  fs={fs} Hz  duration={duration:.0f} s  "
          f"events={len(events)}  (t range {events.min():.0f}-{events.max():.0f} s)")

    # --- baseline pipeline (defaults reproduce the notebook) --------------
    trace, info = preprocess(signal, control, time_s, fs, return_info=True)
    print(f"\nmotion slope = {info['slope']:.3f}   R^2 = {info['r2']:.3f}   "
          f"(compare to notebook cell 22)")
    print(f"dF/F: mean={trace.mean():.3f}  std={trace.std():.3f}  "
          f"min={trace.min():.2f}  max={trace.max():.2f}")

    # --- downstream metric ------------------------------------------------
    peak, t_win, mean_resp, W = peri_event_peak(trace, time_s, events)
    print(f"\nperi-event dF/F peak = {peak:.3f}%   "
          f"(averaged over {len(W)} events, baseline-subtracted)")

    # --- plot the event-average -------------------------------------------
    sem = W.std(axis=0) / np.sqrt(len(W))
    plt.figure(figsize=(7, 4))
    plt.axvline(0, color="k", lw=0.8, ls="--")
    plt.plot(t_win, mean_resp, "g")
    plt.fill_between(t_win, mean_resp - sem, mean_resp + sem, color="g", alpha=0.2)
    plt.xlabel("Time from reward cue (s)")
    plt.ylabel("dF/F (%)")
    plt.title("Baseline pipeline: peri-event average")
    plt.tight_layout()
    out = os.path.join(HERE, "baseline_peri_event.png")
    plt.savefig(out, dpi=120)
    print(f"\nsaved plot -> {out}")


if __name__ == "__main__":
    main()
