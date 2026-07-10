<p align="center">
  <img src="logo.svg" alt="fiberqc" width="440">
</p>
# fiberqc

**Quality control for fiber photometry: is your result real, or an artifact of the preprocessing choices you didn't know you were making?**

Every fiber photometry analysis makes a chain of preprocessing decisions — filter cutoff, bleaching correction, motion regression, normalization. Each is defensible, each is somewhat arbitrary, and a different choice can change the answer. Almost nobody checks. The field's own reference primer (Simpson, Akam, Patriarchi et al., *Neuron* 2024) explicitly notes that there is no systematic comparison of these choices and no established best practice.

`fiberqc` runs your analysis through the whole space of reasonable pipelines at once and tells you whether your conclusion survives — a robustness check you can put in front of a reviewer.

---

## What it does

Given a recording and an event of interest, `fiberqc`:

1. Runs the same downstream analysis through **16 preprocessing pipelines** (every combination of the choices below).
2. Tests whether the effect holds in each.
3. Returns a **verdict** — *robust* (holds everywhere) or *choice-sensitive* (depends on the pipeline) — with a specification curve and an auditable evidence bundle.
4. Uses **Claude** to turn the numbers into a plain-language answer, a publication-ready methods paragraph, and a robustness statement for reviewers.

The multiverse axes (the choices the primer actually discusses):

| Axis | Options |
|------|---------|
| Low-pass filter | 2 Hz / 10 Hz (2nd-order zero-phase Butterworth) |
| Bleaching correction | double-exponential subtraction / 0.001 Hz high-pass |
| Motion correction | OLS / robust (IRLS) regression of the control channel |
| Normalization | dF/F / z-score |

---

## Why it's trustworthy

Nothing is taken on trust. `evidence.py` writes, for every pipeline, its peri-event plot and the **raw per-event amplitudes** that go into the statistic, plus a manifest. Anyone can open a `*_amplitudes.csv`, run their own t-test, and reproduce the exact number reported.

It also doesn't just rubber-stamp "robust." Tested across two independent labs:

- **Clean recordings** (Akam NAc/dLight; Lerner-lab GuPPy dLight) → **robust**: the reward response survives all 16 pipelines.
- **An artifact-heavy recording** (GuPPy artifact tank) → **choice-sensitive**: the preprocessing choice measurably changes the cleaned signal (cross-pipeline CV ≈ 11%), exactly where OLS and robust regression diverge on motion.

Two labs, two sensors, two acquisition systems (pyPhotometry 130 Hz, TDT 1017 Hz), two verdicts — the tool reads each case correctly.

---

## Quickstart

```bash
conda env create -f environment.yml
conda activate fp-robust
pip install -e .                        # makes `import fiberqc` available
export ANTHROPIC_API_KEY=sk-ant-...     # for the Claude interpretation layer
```

**As a library** (the main interface):

```python
import fiberqc as fqc

rec = fqc.load("recording.ppd", events="events.npy")
result = fqc.multiverse(rec)

print(result)                    # MultiverseResult('...': ROBUST, 16/16 significant, t=20.1-20.3)
result.verdict                   # "ROBUST" | "FLIP" | "NULL"
result.spec_curve("spec.png")    # specification curve
result.evidence("out/")          # per-pipeline plots + raw amplitudes + manifest
print(result.report())           # Claude: verdict + methods paragraph + reviewer statement
print(result.ask("Should I use OLS or robust motion correction?"))

# recordings without events (e.g. artifact tanks):
fqc.motion_robustness(rec)       # event-free robustness verdict
```

**As a command line** (natural-language and batch):

```bash
python ask.py --data rec.ppd --events ev.npy \
              --question "Is my reward response real, or a preprocessing artifact?"

python batch.py datasets/        # every recording in a folder -> batch_summary.csv
```

Recordings can be pyPhotometry `.ppd` or any two-channel CSV (signal + control). Converters for GuPPy (`guppy_to_csv.py`) and TDT tanks (`tdt_to_csv.py`) are included.

---

## Built with Claude

The interpretation layer is where Claude sits *inside* the tool. All statistics are computed in Python and handed to Claude as fixed numbers — Claude never invents a value; it writes the narrative around them. It produces the plain-language verdict, the methods paragraph, and the reviewer-facing robustness statement, and answers follow-up questions about the data through `ask.py`. This is what lets a bench scientist drive the whole analysis in natural language, without touching preprocessing code.

---

## Data sources

All validation uses public, published data:

- **Akam `photometry_preprocessing`** — NAc dLight + TdTomato reward-task recording, from Blanco-Pozo et al. (2023). https://github.com/ThomasAkam/photometry_preprocessing
- **GuPPy sample data (Lerner lab)** — dLight NAc Pavlovian reward recordings (clean + artifact), from Sherathiya et al., *Sci Rep* (2021). https://github.com/LernerLab/GuPPy

---

## Honest limitations

- The multiverse **axes** are the preprocessing decisions the reference primer discusses; the specific **values** (e.g. 2 vs 10 Hz) are reasonable defaults, not a measured census of what the field most commonly uses. The axes are user-configurable in `AXES`.
- The clean-vs-artifact comparison uses two metrics: a peri-event t-test when event times exist, and an event-free motion-residual metric when they don't (the artifact tank has no events). Same idea, different statistic — the tool applies whichever fits the data.
- Abstracts do not report preprocessing choices in enough detail to census the literature automatically (measured directly: 0/91 abstracts reported bleaching method). A proper field-wide characterization would need full-text methods sections — a natural next step, and the kind of systematic comparison the primer calls a valuable contribution.

## Roadmap

- User-defined multiverse axes from the command line.
- Scale the engine from single-recording QC to a literature-wide characterization of preprocessing sensitivity.
