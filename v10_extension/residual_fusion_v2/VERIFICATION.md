# Verification boundaries

Local host: macOS, bundled Python3 with NumPy; Torch unavailable. No server task
started. Real pretrained B4/CLIP/VAE/COCO, CUDA, gradient and recovery performance
cannot be confirmed locally. The server launcher requires test_model.py (8 real
Torch tests), existing optical DSP regressions and a bounded8-pair actual smoke.
Missing Torch is a hard error, not a pass or skipped requirement.

Local test_core.py:11 NumPy/stdlib cases; test_report.py:6 raw diagnosis/gate
cases; test_integration.py:6 synthetic complete/diagnostic-stop collection and
receipt/energy/time/artifact negative cases. Includes immutable budget, exact off baseline, shared received hashes,
actual caption invariance, deterministic paired augmentation/distinct donors,
image-cluster statistics and false EOS/score/gate/initialization rejection.
Source Python3.8 grammar, launcher help, release archive restore/additive install,
reinstall/conflict/dependency refusal and recoverable HANDOFF append are checked
by build_release.py. Results live in releases/v10_residual_fusion_v2_build_verification.json.

The report verifier independently recomputes generated WER/quality-gate decision,
checks raw training histories, required parameter updates/gradient coverage,
equal residual scale/init, source/backbone immutability, fresh fixed-payload
fidelity raw measurements, whole-matrix same-physical-receipt controls, exact
off/zero-text baseline, real different-image donor hashes, all energy/time/events,
artifact/source/checkpoint hashes and diagnostic-stop versus main-complete state.
Large checkpoints remain on server; returned lightweight archives contain
auditable server evidence, not a local rerun of those checkpoints.

Successful overfit is a training-only capability diagnostic, not generalization
or a production language decoder. New main arms optimize image residuals only;
main text output is intentionally unchanged B4. Source-joint coding, optimal
image-only/digital comparisons, larger validation and new weather/training
replicates are NOT implemented in this scope and require a separate decision.
