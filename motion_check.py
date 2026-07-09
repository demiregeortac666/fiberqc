"""
motion_check.py
---------------
Event-free robustness metric, for recordings without event times (e.g. the TDT
artifact tank). The concern in artifact-heavy data is motion, and you don't need
task events to see motion's effect: you just ask how much the *preprocessing
choice* changes the cleaned signal.

For each pipeline we compute the cleaned signal and summarize its residual
variability (std of the motion-corrected, detrended signal). If the choice of
OLS vs robust control regression (or bleaching / filtering) barely moves it, the
recording is robust; if it swings a lot, the recording is choice-sensitive --
exactly what you'd expect when big motion artifacts are present, because OLS
chases them and robust regression down-weights them.

Metric: coefficient of variation of residual std across the 16 pipelines.
  low  -> robust (cleaning is insensitive to the choices)
  high -> sensitive (a real motion artifact is being handled differently)

Usage:
    python motion_check.py datasets/guppy_artifact.csv
"""

import sys
import itertools
import numpy as np

from pipeline import preprocess
from data_io import load_recording

AXES = {
    "low_pass":      [10.0, 2.0],
    "bleaching":     ["double_exp", "highpass"],
    "motion":        ["OLS", "robust"],
    "normalization": ["dFF"],          # normalization doesn't affect residual shape here
}


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python motion_check.py <recording.csv>")
    path = sys.argv[1]

    rec = load_recording(path)
    signal, control = rec["signal"], rec["control"]
    time_s, fs = rec["time_s"], rec["fs"]

    keys = list(AXES)
    rows = []
    for combo in itertools.product(*AXES.values()):
        params = dict(zip(keys, combo))
        trace = preprocess(signal, control, time_s, fs, **params)
        rows.append((params, float(np.std(trace))))

    stds = np.array([s for _, s in rows])
    cv = stds.std() / stds.mean()          # coefficient of variation across pipelines

    print(f"{path}: {len(stds)} pipelines")
    for params, s in sorted(rows, key=lambda r: r[1]):
        print(f"  residual std={s:8.3f}   "
              f"{params['motion']:>6} / {params['bleaching']} / lp{params['low_pass']:g}")

    # compare OLS vs robust directly (the motion-sensitive axis)
    ols  = np.mean([s for p, s in rows if p["motion"] == "OLS"])
    rob  = np.mean([s for p, s in rows if p["motion"] == "robust"])
    print(f"\nmean residual std  OLS={ols:.3f}  robust={rob:.3f}  "
          f"(ratio {ols/rob:.2f}x)")
    print(f"cross-pipeline CV = {cv*100:.1f}%   -> "
          f"{'CHOICE-SENSITIVE (motion matters)' if cv > 0.1 else 'robust (cleaning stable)'}")


if __name__ == "__main__":
    main()
