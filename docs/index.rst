.. title:: fiberqc

.. raw:: html

   <p align="center">
     <img src="_static/logo-animated.svg" alt="fiberqc" width="440">
   </p>

**Quality control for fiber photometry: is your result real, or an artifact of
the preprocessing choices you didn't know you were making?**

``fiberqc`` runs your analysis through the whole space of reasonable
preprocessing pipelines at once and tells you whether your conclusion survives —
a robustness check you can put in front of a reviewer.

.. code-block:: python

   import fiberqc as fqc

   rec = fqc.load("recording.ppd", events="events.npy")
   result = fqc.multiverse(rec)

   print(result)                 # verdict + significant count
   result.verdict                # "ROBUST" | "FLIP" | "NULL"
   result.spec_curve("spec.png") # specification curve
   result.evidence("out/")       # per-pipeline plots + raw amplitudes + manifest
   print(result.report())        # Claude: verdict + methods paragraph + reviewer statement

.. toctree::
   :maxdepth: 2
   :hidden:

   api

See the :doc:`api` for the full reference.
