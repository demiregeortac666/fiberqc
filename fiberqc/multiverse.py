"""
fiberqc.multiverse
------------------
The multiverse engine and its result object.

    import fiberqc as fqc
    rec = fqc.load("data.ppd", events="events.npy")
    result = fqc.multiverse(rec)
    result.verdict            # "ROBUST" / "FLIP" / "NULL"
    result.spec_curve("spec.png")
    result.evidence("out/")
    print(result.report())    # Claude narrative (needs ANTHROPIC_API_KEY)
"""

import os
import csv
import itertools

import numpy as np
from scipy.stats import ttest_1samp

from .core import preprocess, peri_event, peri_event_amplitudes, DEFAULT_AXES


class MultiverseResult:
    """Outcome of running an analysis across every pipeline (a pandas-like object)."""

    def __init__(self, rows, keys, recording):
        self.rows = rows          # list of dicts: params + effect, t, p, sig
        self.keys = keys          # the axis names
        self.recording = recording

    # -- summaries ---------------------------------------------------------
    @property
    def n(self):
        return len(self.rows)

    @property
    def n_significant(self):
        return sum(r["sig"] for r in self.rows)

    @property
    def verdict(self):
        if self.n_significant == self.n:
            return "ROBUST"
        if self.n_significant == 0:
            return "NULL"
        return "FLIP"

    @property
    def t_range(self):
        ts = [r["t"] for r in self.rows]
        return (min(ts), max(ts))

    def summary(self):
        lo, hi = self.t_range
        return {"dataset": self.recording.name, "n_pipelines": self.n,
                "n_significant": self.n_significant, "verdict": self.verdict,
                "t_min": round(lo, 2), "t_max": round(hi, 2)}

    def to_df(self):
        import pandas as pd
        return pd.DataFrame(self.rows)

    def to_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
        return path

    def __repr__(self):
        lo, hi = self.t_range
        return (f"MultiverseResult({self.recording.name!r}: {self.verdict}, "
                f"{self.n_significant}/{self.n} significant, t={lo:.1f}-{hi:.1f})")

    # -- specification curve ----------------------------------------------
    def spec_curve(self, path="spec_curve.png"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = sorted(self.rows, key=lambda r: r["t"])
        x = np.arange(len(rows))
        ts = np.array([r["t"] for r in rows])
        sig = np.array([r["sig"] for r in rows])
        options = [(k, v) for k in self.keys for v in sorted({r[k] for r in rows}, key=str)]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 3]})
        ax1.axhline(1.98, color="gray", ls="--", lw=1, label="p=.05")
        ax1.scatter(x[sig], ts[sig], c="#1a7f37", s=40, label="significant", zorder=3)
        ax1.scatter(x[~sig], ts[~sig], c="#cf222e", s=40, label="not significant", zorder=3)
        ax1.set_ylabel("effect (t-statistic)")
        ax1.set_title(f"Specification curve — {self.recording.name} ({self.verdict})")
        ax1.legend(loc="upper left", fontsize=9)

        for row_i, (k, v) in enumerate(options):
            used = np.array([rows[c][k] == v for c in range(len(rows))])
            ax2.scatter(x[used], np.full(used.sum(), row_i), c="#24292f", s=22)
            ax2.scatter(x[~used], np.full((~used).sum(), row_i), c="#d0d7de", s=6)
        ax2.set_yticks(range(len(options)))
        ax2.set_yticklabels([f"{k} = {v}" for k, v in options], fontsize=9)
        ax2.set_xlabel("specification (sorted by effect size)")
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        return path

    # -- auditable evidence bundle ----------------------------------------
    def evidence(self, outdir="variants"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rec = self.recording
        os.makedirs(outdir, exist_ok=True)
        manifest = []
        for r in self.rows:
            params = {k: r[k] for k in self.keys}
            nm = "_".join(f"{v:g}" if isinstance(v, float) else str(v) for v in params.values())
            trace = preprocess(rec.signal, rec.control, rec.time_s, rec.fs, **params)
            t_win, W = peri_event(trace, rec.time_s, rec.events)
            amp = peri_event_amplitudes(trace, rec.time_s, rec.events)

            amp_path = os.path.join(outdir, f"{nm}_amplitudes.csv")
            with open(amp_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["event_index", "amplitude"])
                for i, a in enumerate(amp):
                    w.writerow([i, a])

            png = os.path.join(outdir, f"{nm}_periavg.png")
            mean, sem = W.mean(axis=0), W.std(axis=0) / np.sqrt(len(W))
            plt.figure(figsize=(6, 3.5))
            plt.axvline(0, color="k", lw=0.8, ls="--")
            plt.plot(t_win, mean, "g")
            plt.fill_between(t_win, mean - sem, mean + sem, color="g", alpha=0.2)
            plt.title(f"{nm}\nt={r['t']:.2f} p={r['p']:.1e}", fontsize=9)
            plt.xlabel("Time from event (s)"); plt.ylabel("signal")
            plt.tight_layout(); plt.savefig(png, dpi=110); plt.close()

            manifest.append({**params, "t": round(r["t"], 4), "p": r["p"],
                             "sig": r["sig"], "amplitudes_file": amp_path, "plot_file": png})

        mpath = os.path.join(outdir, "manifest.csv")
        with open(mpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0]))
            w.writeheader(); w.writerows(manifest)
        return outdir

    # -- Claude narrative --------------------------------------------------
    def report(self, path=None, model="claude-sonnet-5"):
        from .report import generate_report
        text = generate_report(self, model=model)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def ask(self, question, model="claude-sonnet-5"):
        """Ask a free-form question about this result in plain language."""
        from .report import answer
        return answer(self, question, model=model)


# ---------------------------------------------------------------- engine
def multiverse(recording, axes=None):
    """Run the analysis through every pipeline. Returns a MultiverseResult."""
    if recording.events is None:
        raise ValueError("This recording has no events; use fiberqc.motion_robustness().")
    axes = axes or DEFAULT_AXES
    keys = list(axes)
    rows = []
    for combo in itertools.product(*axes.values()):
        params = dict(zip(keys, combo))
        trace = preprocess(recording.signal, recording.control,
                           recording.time_s, recording.fs, **params)
        amp = peri_event_amplitudes(trace, recording.time_s, recording.events)
        t, p = ttest_1samp(amp, 0.0)
        rows.append({**params, "effect": float(amp.mean()),
                     "t": float(t), "p": float(p), "sig": bool(p < 0.05)})
    rows.sort(key=lambda r: r["t"])
    return MultiverseResult(rows, keys, recording)


def motion_robustness(recording, axes=None):
    """Event-free robustness: how much does the preprocessing choice change the
    cleaned signal's residual variability? Returns a dict summary."""
    axes = axes or {k: v for k, v in DEFAULT_AXES.items() if k != "normalization"}
    keys = list(axes)
    stds = []
    for combo in itertools.product(*axes.values()):
        params = dict(zip(keys, combo))
        trace = preprocess(recording.signal, recording.control,
                           recording.time_s, recording.fs, **params)
        stds.append((params, float(np.std(trace))))
    vals = np.array([s for _, s in stds])
    cv = vals.std() / vals.mean()
    ols_vals = [s for p, s in stds if p.get("motion") == "OLS"]
    rob_vals = [s for p, s in stds if p.get("motion") == "robust"]
    ols = float(np.mean(ols_vals)) if ols_vals else None
    rob = float(np.mean(rob_vals)) if rob_vals else None
    return {"dataset": recording.name, "n_pipelines": len(stds),
            "cv": float(cv), "ols_std": ols, "robust_std": rob,
            "verdict": "CHOICE-SENSITIVE" if cv > 0.1 else "ROBUST"}
