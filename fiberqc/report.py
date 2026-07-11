"""
fiberqc.report
--------------
The Claude interpretation layer. All statistics are computed in Python and
handed to Claude as fixed numbers; Claude writes the narrative around them and
never invents a value.
"""

from collections import defaultdict

SYSTEM = (
    "You are the analysis assistant inside fiberqc, a fiber photometry robustness tool. "
    "Use ONLY the numbers provided. Never invent a statistic. Be concise, honest, and practical."
)


def _axis_spread(result):
    out = {}
    for k in result.keys:
        by_val = defaultdict(list)
        for r in result.rows:
            by_val[r[k]].append(r["t"])
        means = {v: sum(x) / len(x) for v, x in by_val.items()}
        out[k] = {"means": means, "spread": round(max(means.values()) - min(means.values()), 3)}
    return out


def _prompt(result):
    s = result.summary()
    spreads = _axis_spread(result)
    axis_lines = "\n".join(
        f"  - {k}: t spread {spreads[k]['spread']} "
        f"({', '.join(f'{v}={m:.2f}' for v, m in spreads[k]['means'].items())})"
        for k in result.keys)
    p_max = max(r["p"] for r in result.rows)
    mname = result.metric if isinstance(result.metric, str) else "a custom per-event metric"
    return f"""Interpret this multiverse (specification-curve) analysis for the researcher.

Dataset: {s['dataset']}
Downstream metric: {mname} (response window {result.response} s vs baseline {result.baseline} s), tested per event against zero with a one-sample t-test over {result.rows[0]['n_events']} events.
Specifications: {s['n_pipelines']} pipelines. Significant (p<0.05): {s['n_significant']}/{s['n_pipelines']}. Verdict: {s['verdict']}.
EFFECT SIZE (the magnitude of the finding): raw effect {s['effect_min']:.4g} to {s['effect_max']:.4g} in metric units; Cohen's d {s['d_min']} to {s['d_max']}.
Evidence (t-statistic): {s['t_min']} to {s['t_max']} (largest p-value: {p_max:.1e}).
Do the pipelines agree on the DIRECTION of the effect? {'yes' if s['signs_agree'] else 'NO - some pipelines find a significant increase and others a significant decrease'}.
Note: t = d * sqrt(n), so a large t can come from many events rather than a large effect. Lead with the effect size, treat t as the evidence for it, and say plainly if the magnitude is small.
How much each choice moves the effect (t spread if you flip only that axis):
{axis_lines}

Write markdown with exactly these sections:
## Verdict
2-3 plain-language sentences for the researcher.
## What was tested
1-2 sentences naming the choices and pipeline count.
## Methods paragraph (publication-ready)
One past-tense paragraph suitable to paste into a paper's Methods. Use the metric and windows given above.
## For reviewers
1-2 sentences the researcher can quote, citing the concrete numbers, to show the result is not an artifact of preprocessing choices.
"""


def generate_report(result, model="claude-sonnet-5"):
    from anthropic import Anthropic
    client = Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=1500, system=SYSTEM,
        messages=[{"role": "user", "content": _prompt(result)}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def _results_context(result):
    s = result.summary()
    lines = [", ".join(f"{k}={r[k]}" for k in result.keys) +
             f" -> t={r['t']:.2f}, p={r['p']:.1e}, {'significant' if r['sig'] else 'ns'}"
             for r in result.rows]
    return (f"Dataset {s['dataset']}: {s['n_pipelines']} pipelines, "
            f"{s['n_significant']}/{s['n_pipelines']} significant, verdict {s['verdict']}, "
            f"t {s['t_min']}-{s['t_max']}.\n" + "\n".join(lines))


def answer(result, question, model="claude-sonnet-5"):
    """Answer a free-form question about the result, grounded in the numbers."""
    from anthropic import Anthropic
    client = Anthropic()
    prompt = f"{_results_context(result)}\n\nResearcher asks: {question}"
    resp = client.messages.create(
        model=model, max_tokens=1500, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")
