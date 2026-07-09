"""
ask.py
------
Natural-language interface to FP-Robust. Point it at a recording, ask a question
in plain English, and Claude runs the multiverse and answers -- no code needed.

    python ask.py --data ../photometry_preprocessing/data/m53_NAc_L-2019-11-24-093939.ppd \
                  --events ../photometry_preprocessing/data/reward_cue_times.npy \
                  --question "Is my reward response real or could it be a preprocessing artifact?"

Omit --question for an interactive session (runs the engine once, then chat).
Works with .ppd or any two-channel CSV (see data_io.py).

Setup: export ANTHROPIC_API_KEY=sk-ant-...
"""

import os
import sys
import argparse
import itertools

import numpy as np
from scipy.stats import ttest_1samp
from anthropic import Anthropic

from pipeline import preprocess, peri_event
from data_io import load_recording

MODEL = "claude-sonnet-5"
client = Anthropic()

AXES = {
    "low_pass":      [10.0, 2.0],
    "bleaching":     ["double_exp", "highpass"],
    "motion":        ["OLS", "robust"],
    "normalization": ["dFF", "zscore"],
}
BASELINE_WIN = (-1.0, 0.0)
RESPONSE_WIN = (0.0, 2.0)

SYSTEM = (
    "You are the analysis assistant inside FP-Robust, a fiber photometry robustness tool. "
    "Answer the researcher's questions about their peri-event response using ONLY the multiverse "
    "results provided in the message. Never invent numbers. Be concise, honest, and practical. "
    "If the result holds across all pipelines, say it is robust and what that means. If asked which "
    "preprocessing choice to use, say whether the conclusion actually depends on it."
)


def claude_text(prompt):
    resp = client.messages.create(
        model=MODEL, max_tokens=1500, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def event_amplitudes(trace, time_s, events):
    t_win, W = peri_event(trace, time_s, events)
    base = W[:, (t_win >= BASELINE_WIN[0]) & (t_win < BASELINE_WIN[1])].mean(axis=1)
    resp = W[:, (t_win >= RESPONSE_WIN[0]) & (t_win < RESPONSE_WIN[1])].mean(axis=1)
    return resp - base


def run_multiverse(signal, control, time_s, fs, events):
    keys = list(AXES)
    rows = []
    for combo in itertools.product(*AXES.values()):
        params = dict(zip(keys, combo))
        trace = preprocess(signal, control, time_s, fs, **params)
        amp = event_amplitudes(trace, time_s, events)
        t, p = ttest_1samp(amp, 0.0)
        rows.append({**params, "effect": float(amp.mean()),
                     "t": float(t), "p": float(p), "sig": bool(p < 0.05)})
    rows.sort(key=lambda r: r["t"])
    return keys, rows


def results_context(keys, rows):
    n = len(rows)
    n_sig = sum(r["sig"] for r in rows)
    ts = [r["t"] for r in rows]
    lines = [", ".join(f"{k}={r[k]}" for k in keys) +
             f" -> t={r['t']:.2f}, p={r['p']:.1e}, {'significant' if r['sig'] else 'ns'}"
             for r in rows]
    return (f"Multiverse of {n} preprocessing pipelines. "
            f"Significant (p<0.05): {n_sig}/{n}. t-statistic range {min(ts):.2f}-{max(ts):.2f}.\n"
            + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--fs", type=float)
    ap.add_argument("--question")
    a = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")

    rec = load_recording(a.data, a.events, fs=a.fs)
    signal, control = rec["signal"], rec["control"]
    time_s, fs, events = rec["time_s"], rec["fs"], rec["events"]
    if events.max() > time_s[-1] * 1.5:
        events = events / 1000

    print(f"loaded {len(signal)} samples @ {fs:.0f} Hz, {len(events)} events. "
          f"running 16 pipelines (~1 min)...")
    keys, rows = run_multiverse(signal, control, time_s, fs, events)
    ctx = results_context(keys, rows)
    n_sig = sum(r["sig"] for r in rows)
    print(f"done: {n_sig}/16 pipelines significant.\n")

    q = a.question or input("Ask about your data> ")
    while q and q.strip():
        print("\n" + claude_text(f"{ctx}\n\nResearcher asks: {q}") + "\n")
        if a.question:
            break
        q = input("Ask about your data> ")


if __name__ == "__main__":
    main()
