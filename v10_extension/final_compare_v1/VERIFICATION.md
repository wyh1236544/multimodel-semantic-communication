# Verification boundaries

Executed locally using bundled Python3.11 and NumPy/Pillow (not the server Python):

- 24 final-comparison NumPy/core tests: actual sampled optical transport, all9
  launch points, exact27-repeat budget, FEC/QAM accounting, common pilot/field,
  independent JPEG/text CRC, infeasible power/budget rejection, complete/duplicate/
  missing matrices, image-cluster bootstrap, offline report generation, immutable
  completed hashes and strict collection/unknown-file refusal.
- Python source parsing with Python3.8 grammar.
- Existing physical/digital and optical-link NumPy regression suites.
- Deterministic source-only tar restoration, exact manifest comparison, additive
  synthetic installation/reinstallation, preserved shared-handoff byte backup,
  conflict and missing-dependency refusal. Build report in project releases/.

Not executed locally:

- 5 Torch CPU frame tests (test_model.py), since Torch is unavailable in the
  supplied local runtime. These run first on the user's existing server environment.
- Loading actual original A delta/V10 base/CLIP/VAE; real COCO/GPU1 outputs or
  numerical image/text quality. No mock neural outputs presented as experiments.
- Hardware calibration, nonzero CFO/line widths, live optical measurements,
  confirmatory test split or publication readiness.

The user must run the gated pipeline. A failure preserves logs/partial records,
not a valid COMPLETE. Collect independently rechecks hashes/matrix/resources.
