# Verification boundary — 2026-10-13

Local macOS bundled Python3.12/NumPy only. No Torch/Transformers, saved neural
weights, COCO images or school-GPU execution in this task. No dependencies
installed, global settings changed or server task launched. Never infer learned
performance from parsing, synthetic report fixtures or NumPy optical tests.

## Local checks

- New test_core.py8: strict budgets/config, block-isolated source attention,
  equal6144-real/3072-complex geometry, real new full-frame NumPy V3 waveforms,
  source split safety and quality gates.
- New test_report.py12: negative text-prefit completion, no TRAIN8 init,
  metric/update/source drift, missing/duplicate control, blocked joint/noise
  expansion, full positive synthetic three-arm matrix, registered-size mismatch,
  missing optical event and result archive roundtrip/refused overwrite.
  Synthetic positives are verifier fixtures, NOT actual trained network evidence.
- Unchanged V4 test_core10/test_report11 and COCO modal V2 core11/report10
  regressions rerun locally. New plus these legacy suites:62 tests.
- New Python sources parse with Python3.8 grammar. Launcher --help tested.
- Source builder enforces exact USTAR member/file restore, install/reinstall,
  source prerequisite hashes, original shared handoff backup, source conflict,
  prerequisite drift and symlink rejection. Actual release/build result is in
  releases/v10_clip_bart_joint_v5_build_verification.json outside source archive.

Commands use the bundled Python with -B from the local project root. Extraction
deprecation warnings under3.12 do not change successful exit codes; source tar
contains only prevalidated known regular-file members.

## Mandatory server gates, not locally executed

10 new test_model.py Torch/HF tiny-model tests: V4 private text identity at
initialization; separate source and receiver modality independence; no-CLIP-text
source invariance; equal registered bridge count and group power; no receiver
oracle/erasure; real group gradients and updates/frozen preservation; grad/no-grad
numeric consistency; warm freezing and checkpoint state restore.

Real saved-model smoke follows: actual BART cache/incremental consistency;
full compatible base/A/B2/B4/CLIP/VAE identity; frozen VAE input gradients;
new registered C/I/T groups must have nonzero gradients AND actual updates;
real disk reload, FP32 GPU peak/free reserve; actual generated text and pixels.
No skipped failing updates, automatic retries or lowered quality thresholds.

Main: one seed42; source-controlled prefit1024 captions; joint256pairs; fixed
32validation and final checkpoints. Early gates, errors and failed frames remain
visible. Same registered size does not mean same active gradient count or FLOPs.
3072 payload differs from old2304; this is not old-budget bandwidth superiority.
Complete markers certify execution, not joint benefit/hardware/paper efficacy.
