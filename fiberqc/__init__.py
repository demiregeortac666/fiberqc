"""
fiberqc — quality control for fiber photometry.

Is your result real, or an artifact of the preprocessing choices you didn't
know you were making? fiberqc runs your analysis through the whole space of
reasonable pipelines and tells you whether your conclusion survives.

    import fiberqc as fqc

    rec = fqc.load("data.ppd", events="events.npy")
    result = fqc.multiverse(rec)
    print(result)                 # verdict + significant count
    result.spec_curve("spec.png")
    result.evidence("out/")
    print(result.report())        # Claude narrative (needs ANTHROPIC_API_KEY)
"""

from .core import Recording, preprocess, peri_event, peri_event_amplitudes, DEFAULT_AXES as AXES
from .io import load, load_events
from .multiverse import multiverse, motion_robustness, MultiverseResult

__version__ = "0.1.0"
__all__ = [
    "load", "load_events", "multiverse", "motion_robustness",
    "preprocess", "peri_event", "peri_event_amplitudes",
    "Recording", "MultiverseResult", "AXES",
]
