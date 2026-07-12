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
    """Outcome of running an analysis across every preprocessing pipeline.

    Returned by :func:`fiberqc.multiverse` — you rarely build it yourself.
    Inspect it with :attr:`verdict`, turn it into a table with :meth:`to_df`,
    plot it with :meth:`spec_curve`, dump an auditable bundle with
    :meth:`evidence`, or get a written interpretation with :meth:`report` and
    :meth:`ask`.

    Attributes
    ----------
    verdict : str
        ``"ROBUST"`` (significant in every pipeline), ``"NULL"`` (in none), or
        ``"FLIP"`` (in some but not all).
    rows : list of dict
        One record per pipeline: its parameters plus ``effect``, ``t``, ``p``, ``sig``.
    keys : list of str
        The axis names that were varied.
    """

    def __init__(self, rows, keys, recording, *, my_pipeline=None, metric="mean",
                 baseline=(-1.0, 0.0), response=(0.0, 2.0), pre=1.0, post=3.0):
        self.my_pipeline = my_pipeline
        self.rows = rows          # list of dicts: params + effect, t, p, sig
        self.keys = keys          # the axis names
        self.recording = recording
        self.metric = metric      # name (or callable) of the downstream metric used
        self.baseline = baseline
        self.response = response
        self.pre, self.post = pre, post

    # -- summaries ---------------------------------------------------------
    @property
    def n(self):
        return len(self.rows)

    @property
    def n_significant(self):
        return sum(r["sig"] for r in self.rows)

    @property
    def signs_agree(self):
        """Do all significant pipelines agree on the direction of the effect?

        A specification where half the pipelines find a significant increase and
        half a significant decrease is maximally choice-sensitive, not robust —
        so the sign has to be part of the verdict, not just the p-value.
        """
        signs = {np.sign(r["t"]) for r in self.rows if r["sig"]}
        return len(signs) <= 1

    @property
    def verdict(self):
        if self.n_significant == 0:
            return "NULL"
        if not self.signs_agree:
            return "FLIP"          # significant in both directions
        if self.n_significant == self.n:
            return "ROBUST"
        return "FLIP"              # significant in some pipelines, not others

    @property
    def t_range(self):
        ts = [r["t"] for r in self.rows]
        return (min(ts), max(ts))

    @property
    def effect_range(self):
        es = [r["effect"] for r in self.rows]
        return (min(es), max(es))

    @property
    def my_row(self):
        """The specification the researcher actually ran, if they declared one."""
        for r in self.rows:
            if r.get("mine"):
                return r
        return None

    def where_do_i_stand(self):
        """How the researcher's own pipeline compares with every alternative.

        A multiverse that only says "the choice matters" is not actionable. What a
        researcher needs is: *my* pipeline says this; of the reasonable
        alternatives, this many agree with me, and this many do not.

        Returns a dict, or None if no ``my_pipeline`` was declared.
        """
        mine = self.my_row
        if mine is None:
            return None
        others = [r for r in self.rows if not r.get("mine")]
        same_sign = [r for r in others
                     if r["sig"] and np.sign(r["d"]) == np.sign(mine["d"])]
        opposite = [r for r in others
                    if r["sig"] and np.sign(r["d"]) != np.sign(mine["d"])]
        null = [r for r in others if not r["sig"]]
        return {
            "my_effect": mine["effect"],
            "my_d": mine["d"],
            "my_t": mine["t"],
            "my_p": mine["p"],
            "my_significant": mine["sig"],
            "n_alternatives": len(others),
            "agree": len(same_sign),
            "opposite_sign": len(opposite),
            "not_significant": len(null),
            "d_if_i_had_chosen_otherwise": (min(r["d"] for r in others),
                                            max(r["d"] for r in others)) if others else None,
        }

    @property
    def culprit(self):
        """The axis whose choice moves the effect size most — the knob that
        actually decides your result. Returns (axis_name, spread_in_d)."""
        best, best_spread = None, -1.0
        for k in self.keys:
            by_val = {}
            for r in self.rows:
                by_val.setdefault(r[k], []).append(r["d"])
            means = [np.mean(v) for v in by_val.values()]
            spread = float(max(means) - min(means)) if len(means) > 1 else 0.0
            if spread > best_spread:
                best, best_spread = k, spread
        return best, best_spread

    @property
    def d_range(self):
        ds = [r["d"] for r in self.rows if not np.isnan(r["d"])]
        return (min(ds), max(ds)) if ds else (np.nan, np.nan)

    def summary(self):
        lo, hi = self.t_range
        elo, ehi = self.effect_range
        dlo, dhi = self.d_range
        return {"dataset": self.recording.name, "n_pipelines": self.n,
                "n_significant": self.n_significant, "verdict": self.verdict,
                "signs_agree": self.signs_agree,
                "effect_min": elo, "effect_max": ehi,
                "d_min": round(dlo, 3), "d_max": round(dhi, 3),
                "t_min": round(lo, 2), "t_max": round(hi, 2)}

    def to_df(self):
        """Every pipeline as a table, ordered by effect size.

        The axis that decides the result comes first, so the split is visible in
        the table itself rather than only in the figure. ``my_pipeline`` marks the
        specification the researcher declared.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd

        df = pd.DataFrame(self.rows).rename(columns={"mine": "my_pipeline"})
        if "my_pipeline" not in df.columns:
            df["my_pipeline"] = False

        culprit_axis, _ = self.culprit
        rest = [k for k in self.keys if k != culprit_axis]
        stats = ["effect", "d", "t", "p", "sig", "n_events"]
        order = ([culprit_axis] + rest
                 + [c for c in stats if c in df.columns] + ["my_pipeline"])
        return df[order].sort_values("d").reset_index(drop=True)

    def to_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
        return path

    def __repr__(self):
        lo, hi = self.t_range
        dlo, dhi = self.d_range
        warn = "" if self.signs_agree else ", SIGN FLIPS"
        mine = self.my_row
        anchor = f", yours d={mine['d']:.2f}" if mine else ""
        return (f"MultiverseResult({self.recording.name!r}: {self.verdict}, "
                f"{self.n_significant}/{self.n} significant, "
                f"d={dlo:.2f}..{dhi:.2f}{anchor}, t={lo:.1f}..{hi:.1f}{warn})")

    # -- specification curve ----------------------------------------------
    def spec_curve(self, path="spec_curve.png", dpi=200, figsize=(9, 9)):
        """Save a publication-quality specification curve.

        The top panel is the **effect size**, not the t-statistic. A t mixes
        magnitude with variability and grows with the square root of the event
        count, so a trivially small effect measured over many events can look
        overwhelming (Chen et al., 2017). The t-statistic is shown separately, as
        the evidence *for* the effect rather than as the effect.

        The bottom panel shows which choices produced each specification. The
        axis that moves the result most is highlighted — that is the knob your
        conclusion actually hangs on.

        Parameters
        ----------
        path : str
            Output image path (``.png``, ``.pdf`` and ``.svg`` all work).
        dpi : int, optional
            Resolution. 200 is fine for a figure panel; use 300+ for print.
        figsize : tuple, optional
            Figure size in inches.

        Returns
        -------
        str
            The path written.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        POS   = "#1b6ca8"   # effect in one direction
        NEG   = "#d1495b"   # effect in the other
        NS    = "#b8bfc7"   # not significant
        INK   = "#2b2f33"
        FAINT = "#e3e6e9"
        HL    = "#e07a20"   # the axis that decides the result

        rows = sorted(self.rows, key=lambda r: r["d"])
        x = np.arange(len(rows))
        ds = np.array([r["d"] for r in rows])
        ts = np.array([r["t"] for r in rows])
        sig = np.array([r["sig"] for r in rows])
        culprit_axis, spread = self.culprit

        colours = [NS if not s else (POS if d >= 0 else NEG)
                   for s, d in zip(sig, ds)]

        options = [(k, v) for k in self.keys
                   for v in sorted({r[k] for r in rows}, key=str)]

        fig, (ax0, ax1, ax2) = plt.subplots(
            3, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [3, 1.3, 2.6], "hspace": 0.12})

        for ax in (ax0, ax1, ax2):
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(INK)
            ax.tick_params(colors=INK, labelsize=9)

        # ---- effect size: the headline
        ax0.axhline(0, color=INK, lw=1, zorder=1)
        ax0.grid(axis="y", color=FAINT, lw=0.7, zorder=0)
        ax0.set_axisbelow(True)
        ax0.scatter(x, ds, c=colours, s=54, zorder=3,
                    edgecolors="white", linewidths=0.8)

        mine_i = [i for i, r in enumerate(rows) if r.get("mine")]
        if mine_i:
            i = mine_i[0]
            ax0.axvline(i, color=INK, lw=1, ls=(0, (2, 2)), zorder=1, alpha=0.55)
            ax0.scatter([i], [ds[i]], s=190, facecolors="none",
                        edgecolors=INK, linewidths=1.6, zorder=4)
            ax0.annotate("your pipeline", xy=(i, ds[i]),
                         xytext=(0, 16), textcoords="offset points",
                         ha="center", fontsize=8.5, color=INK, fontweight="bold")
        ax0.set_ylabel("effect size  (Cohen's d)", fontsize=10, color=INK)

        handles = [Line2D([], [], marker="o", ls="", mfc=POS, mec="white", ms=8,
                          label="significant, positive"),
                   Line2D([], [], marker="o", ls="", mfc=NEG, mec="white", ms=8,
                          label="significant, negative"),
                   Line2D([], [], marker="o", ls="", mfc=NS, mec="white", ms=8,
                          label="not significant")]
        ax0.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.17),
                   ncol=3, frameon=False, fontsize=8.5, handletextpad=.3,
                   columnspacing=1.6)
        ax0.margins(y=0.16)          # keep points clear of the frame

        title = f"{self.recording.name}  —  {self.verdict}"
        ax0.set_title(title, fontsize=12.5, color=INK, loc="left", pad=34,
                      fontweight="bold")

        # One headline, and only the thing that should alarm you is in alarm colour.
        stand = self.where_do_i_stand()
        if stand and stand["opposite_sign"]:
            head = (f"{stand['opposite_sign']} of {stand['n_alternatives']} reasonable "
                    f"alternatives reverse your result")
            head_c = NEG
        elif not self.signs_agree:
            head = "pipelines disagree on the direction of the effect"
            head_c = NEG
        elif stand:
            head = (f"{stand['agree']}/{stand['n_alternatives']} alternatives agree "
                    f"with your pipeline")
            head_c = "#5b6470"
        else:
            head = f"{self.n_significant}/{self.n} specifications significant"
            head_c = "#5b6470"
        ax0.annotate(head, xy=(0, 1.105), xycoords="axes fraction",
                     fontsize=9.5, color=head_c, fontweight="bold")

        # Quiet context, well out of the way.
        ctx = (f"{self.n_significant}/{self.n} significant   ·   "
               f"the choice that decides it: {culprit_axis}")
        ax0.annotate(ctx, xy=(0, 1.035), xycoords="axes fraction",
                     fontsize=8.5, color="#8a929b")

        # ---- evidence, kept apart from magnitude
        ax1.axhline(0, color=INK, lw=0.9)
        for y in (1.98, -1.98):
            ax1.axhline(y, color=NS, ls=(0, (4, 3)), lw=0.8)
        ax1.scatter(x, ts, c=colours, s=26, zorder=3,
                    edgecolors="white", linewidths=0.6)
        ax1.set_ylabel("evidence\n(t-statistic)", fontsize=9, color="#5b6470")
        ax1.annotate("p = .05", xy=(len(x) - 0.5, 2.1), fontsize=7.5,
                     color="#8a929b", ha="right")

        # ---- which choices produced each specification
        for row_i, (k, v) in enumerate(options):
            used = np.array([rows[c][k] == v for c in range(len(rows))])
            on_colour = HL if k == culprit_axis else INK
            size = 30 if k == culprit_axis else 22
            ax2.scatter(x[~used], np.full((~used).sum(), row_i),
                        c=FAINT, s=8, zorder=2)
            ax2.scatter(x[used], np.full(used.sum(), row_i),
                        c=on_colour, s=size, zorder=3)

        labels = []
        for k, v in options:
            lab = f"{k} = {v}"
            labels.append(lab)
        ax2.set_yticks(range(len(options)))
        ax2.set_yticklabels(labels, fontsize=8.5)
        for tick, (k, _) in zip(ax2.get_yticklabels(), options):
            if k == culprit_axis:
                tick.set_color(HL)
                tick.set_fontweight("bold")
        ax2.set_xlabel("specification, ordered by effect size", fontsize=10, color=INK)
        ax2.set_ylim(len(options) - 0.5, -0.5)
        ax2.set_xlim(-0.8, len(x) - 0.2)
        ax2.grid(axis="x", color=FAINT, lw=0.6, zorder=0)
        ax2.set_axisbelow(True)

        fig.align_ylabels([ax0, ax1, ax2])
        plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        return path

    # -- auditable evidence bundle ----------------------------------------
    def evidence(self, outdir="variants"):
        """Write an auditable evidence bundle: for every pipeline, its peri-event
        plot and the raw per-event metric values, plus a ``manifest.csv``.

        Anyone can open a values file, rerun the test themselves, and reproduce
        the exact number reported — the analysis is fully inspectable.

        Parameters
        ----------
        outdir : str
            Directory to write the bundle into (created if needed).

        Returns
        -------
        str
            The output directory.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from . import metrics as _metrics

        rec = self.recording
        os.makedirs(outdir, exist_ok=True)
        mname = self.metric if isinstance(self.metric, str) else "custom"
        manifest = []
        for r in self.rows:
            params = {k: r[k] for k in self.keys}
            nm = "_".join(f"{v:g}" if isinstance(v, float) else str(v) for v in params.values())
            trace = preprocess(rec.signal, rec.control, rec.time_s, rec.fs, **params)
            t_win, W = peri_event(trace, rec.time_s, rec.events, pre=self.pre, post=self.post)
            vals = _metrics.compute(self.metric, trace, rec.time_s, rec.events,
                                    self.baseline, self.response, self.pre, self.post)

            amp_path = os.path.join(outdir, f"{nm}_{mname}.csv")
            with open(amp_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["event_index", mname])
                for i, a in enumerate(vals):
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

            manifest.append({**params, "metric": mname,
                             "effect": r["effect"], "d": round(r["d"], 4),
                             "n_events": r["n_events"], "t": round(r["t"], 4),
                             "p": r["p"], "sig": r["sig"],
                             "values_file": amp_path, "plot_file": png})

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
def multiverse(recording, axes=None, *, my_pipeline=None, metric="mean",
               baseline=(-1.0, 0.0), response=(0.0, 2.0), pre=1.0, post=3.0):
    """Run the analysis across every preprocessing pipeline.

    Builds the multiverse of pipelines (every combination of the values in
    ``axes``), computes the chosen downstream ``metric`` per event in each, and
    tests it against zero. The result reports whether the effect is *robust*
    (holds in every pipeline) or *choice-sensitive*.

    Parameters
    ----------
    recording : Recording
        A recording with events, from :func:`fiberqc.load`.
    axes : dict, optional
        ``{knob: [values]}`` to explore. Supported knobs are the
        :func:`fiberqc.preprocess` arguments: ``low_pass``, ``median_filter``,
        ``bleaching``, ``motion``, ``normalization``. Defaults to
        :data:`fiberqc.AXES` (the four choices the reference primer discusses).
    metric : {"mean", "peak", "auc"} or callable, optional
        Downstream metric. Built-ins compare a response window to a baseline
        window. A callable ``metric(t_win, W) -> per-event array`` lets you
        define your own, where ``W`` is the ``(n_events, n_samples)`` peri-event
        matrix. Return values where 0 means "no effect". Default ``"mean"``.
    baseline, response : tuple of float, optional
        ``(start, stop)`` in seconds for the pre- and post-event windows.
    pre, post : float, optional
        Seconds of trace to align around each event.

    Returns
    -------
    MultiverseResult
        Holds every pipeline's outcome, with ``.verdict``, ``.spec_curve()``,
        ``.evidence()``, ``.report()`` and ``.ask()``.

    Examples
    --------
    >>> result = fqc.multiverse(rec)
    >>> result.verdict
    'ROBUST'
    >>> result = fqc.multiverse(rec, metric="peak", response=(0, 1))
    >>> result = fqc.multiverse(rec, axes={"low_pass": [1, 5, 10], "motion": ["OLS", "robust"]})
    """
    from . import metrics as _metrics
    if recording.events is None:
        raise ValueError("This recording has no events; use fiberqc.motion_robustness().")
    axes = dict(axes or DEFAULT_AXES)
    keys = list(axes)

    # The pipeline the researcher actually ran is the anchor: without it, a
    # multiverse says "the choice matters" but never "and here is where YOUR
    # choice lands". Fold its values into the axes so it is one of the
    # specifications, then mark it in the result.
    if my_pipeline:
        unknown = set(my_pipeline) - set(keys)
        if unknown:
            raise ValueError(f"my_pipeline has knobs that are not in axes: {sorted(unknown)}. "
                             f"Add them to axes, or use the same names: {keys}")
        for k, v in my_pipeline.items():
            if v not in axes[k]:
                axes[k] = list(axes[k]) + [v]

    combos = list(itertools.product(*axes.values()))
    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        trace = preprocess(recording.signal, recording.control,
                           recording.time_s, recording.fs, **params)
        vals = _metrics.compute(metric, trace, recording.time_s, recording.events,
                                baseline, response, pre, post)
        t, p = ttest_1samp(vals, 0.0)
        sd = float(np.std(vals, ddof=1))
        mean = float(np.mean(vals))
        # Cohen's d: the magnitude in units of between-event variability. Unlike t
        # it does not grow with the number of events, so a tiny effect measured
        # over many trials cannot masquerade as a strong one.
        d = mean / sd if sd > 0 else np.nan
        is_mine = bool(my_pipeline) and all(params[k] == v for k, v in my_pipeline.items())
        rows.append({**params, "effect": mean, "d": float(d), "n_events": int(len(vals)),
                     "t": float(t), "p": float(p), "sig": bool(p < 0.05),
                     "mine": is_mine})
    rows.sort(key=lambda r: r["t"])
    return MultiverseResult(rows, keys, recording, my_pipeline=my_pipeline, metric=metric,
                            baseline=baseline, response=response, pre=pre, post=post)


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
