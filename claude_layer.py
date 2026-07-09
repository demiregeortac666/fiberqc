"""
claude_layer.py
---------------
The "Built with Claude" core: turn the multiverse numbers into science.

Reads multiverse_results.csv, computes all the statistics in Python (so no
number is ever invented by the model), then asks Claude to write:
  1. a plain-language verdict,
  2. a publication-ready methods paragraph,
  3. a reviewer-facing robustness statement.

Principle: numbers come from Python, prose comes from Claude.

Setup:  export ANTHROPIC_API_KEY=sk-ant-...   (personal account with the credit)
Output: robustness_report.md
"""

import os
import csv
import sys
from collections import defaultdict

from anthropic import Anthropic

MODEL   = "claude-sonnet-5"          # generative writing -> Sonnet, not Haiku
RESULTS = "multiverse_results.csv"
OUT     = "robustness_report.md"

client = Anthropic()

AXES = ["low_pass", "bleaching", "motion", "normalization"]

# Factual context so the methods paragraph is accurate. Edit if the dataset changes.
DATASET = (
    "Recording: mouse nucleus accumbens, dLight dopamine sensor (signal channel) "
    "with a co-expressed TdTomato control channel, during a reward-guided task. "
    "Sampling rate 130 Hz, ~90 min, 137 reward-cue events. "
    "Downstream analysis: peri-event response to the reward cue, quantified as the "
    "per-event amplitude = mean dF/F in the 0-2 s response window minus the mean in "
    "the -1-0 s baseline window, tested against zero with a one-sample t-test across events."
)
METHODS_AXES = (
    "Preprocessing choices varied (multiverse): "
    "low-pass cutoff 2 vs 10 Hz (2nd-order Butterworth, zero-phase filtfilt); "
    "bleaching correction by double-exponential subtraction vs 0.001 Hz high-pass filter; "
    "motion correction by regressing the TdTomato control onto the signal using OLS vs "
    "robust/IRLS (Huber) regression; normalization as dF/F vs z-score. "
    "All 2x2x2x2 = 16 combinations were run."
)


def load_results():
    if not os.path.exists(RESULTS):
        sys.exit(f"{RESULTS} not found -- run multiverse.py first.")
    rows = []
    with open(RESULTS) as f:
        for r in csv.DictReader(f):
            r["t"] = float(r["t"]); r["p"] = float(r["p"])
            r["effect"] = float(r["effect"]); r["sig"] = r["sig"] == "True"
            rows.append(r)
    return rows


def summarize(rows):
    n = len(rows)
    n_sig = sum(r["sig"] for r in rows)
    ts = [r["t"] for r in rows]
    # which axis moves the effect (t) most, holding it as the only difference
    axis_effect = {}
    for ax in AXES:
        by_val = defaultdict(list)
        for r in rows:
            by_val[r[ax]].append(r["t"])
        means = {v: sum(x) / len(x) for v, x in by_val.items()}
        vals = list(means)
        axis_effect[ax] = {"means": means,
                           "spread": round(max(means.values()) - min(means.values()), 3)}
    # effect ranges per normalization (units differ: dFF % vs z-score sd)
    eff = defaultdict(list)
    for r in rows:
        eff[r["normalization"]].append(r["effect"])
    return {
        "n": n, "n_sig": n_sig, "n_notsig": n - n_sig,
        "t_min": round(min(ts), 2), "t_max": round(max(ts), 2),
        "p_max": max(r["p"] for r in rows),
        "axis_effect": axis_effect,
        "effect_ranges": {k: (round(min(v), 3), round(max(v), 3)) for k, v in eff.items()},
    }


def build_prompt(rows, s):
    axis_lines = "\n".join(
        f"  - {ax}: t spread {s['axis_effect'][ax]['spread']} "
        f"({', '.join(f'{v}={m:.2f}' for v, m in s['axis_effect'][ax]['means'].items())})"
        for ax in AXES)
    verdict = ("ROBUST (significant in every pipeline)" if s["n_sig"] == s["n"]
               else "FLIPS" if s["n_sig"] else "NULL in every pipeline")
    return f"""You are the analysis assistant inside FP-Robust, a fiber photometry robustness tool.
Write a short report interpreting a multiverse (specification-curve) analysis for the researcher.

Use ONLY the numbers provided below. Do not invent any statistic. Be precise and honest; if
the result is robust, say so plainly and explain what that means for the researcher.

=== DATASET ===
{DATASET}

=== {METHODS_AXES}

=== RESULTS ===
Specifications: {s['n']} pipelines. Significant (p<0.05): {s['n_sig']}/{s['n']}. Verdict: {verdict}.
t-statistic range across all pipelines: {s['t_min']} to {s['t_max']} (largest p-value: {s['p_max']:.1e}).
Effect magnitude range: dF/F variants {s['effect_ranges'].get('dFF')} (% dF/F units), z-score variants {s['effect_ranges'].get('zscore')} (sd units).
How much each choice moves the effect (t spread if you flip only that axis):
{axis_lines}

=== WRITE THE REPORT IN MARKDOWN WITH EXACTLY THESE SECTIONS ===
## Verdict
2-3 sentences, plain language, for the researcher.

## What was tested
1-2 sentences naming the choices explored and how many pipelines.

## Methods paragraph (publication-ready)
One paragraph, past tense, accurate to the pipeline described above, suitable to paste into a paper's Methods.

## For reviewers
1-2 sentences the researcher can quote to demonstrate the result is not an artifact of preprocessing choices, citing the concrete numbers.
"""


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")
    rows = load_results()
    s = summarize(rows)
    resp = client.messages.create(
        model=MODEL, max_tokens=1500,
        messages=[{"role": "user", "content": build_prompt(rows, s)}],
    )
    report = report = "".join(b.text for b in resp.content if b.type == "text")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n---\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
