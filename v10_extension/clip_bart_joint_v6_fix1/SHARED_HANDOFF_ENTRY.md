## 2026-10-13 — CLIP/BART joint V6 fix1: clean SNR verification

User's real GPU smoke executed all three arms and both evaluated conditions, then
finish rejected receiver-condition NPZ SNR. GPU free28.27GiB; no OOM in supplied
log. Unchanged V5 clean_args supplies40dB as FINITE CLEAN CONDITIONING SENTINEL,
not a measured physical/pilot SNR. V6 evaluator faithfully saves it, but report
expected20dB. Original synthetic fixture duplicated wrong20 and hid mismatch.

User-authorized report-only repair: expected clean40, retain finite checks and
1e-4 physical float32 tolerance; FSO still compares actual recorded pilot SNR.
Error now identifies condition/image/expected/stored. No receiver/model/source/
training/schedule/loss/resource/FSO parameters or quality thresholds changed.
32 existing reports +7 new cases: production clean_args body NumPy shim and
actual Torch interface, clean20/zero rejection, physical float32 match/mismatch,
nonfinite rejection. Local38 pass,1 real-Torch case skipped (Torch unavailable);
server must run that case plus existing26 Torch/HF tests and new real smoke.
New regression reproduced original failure BEFORE fix; full positive clean/
paired-FSO matrices passed AFTER fix. This is not proof of improved recovery.

Hash-locked fix1 installer preflights all replacements/unchanged module and378
historical sources, backs up report/test/manifest/HANDOFF before replacement,
rejects unknown/mixed/symlink/drift, idempotent. Old V6 archive and failed
experiment_results/v10_clip_bart_joint_v6_01 remain unchanged; do not delete
FAILED or reseal old results. New run v10_clip_bart_joint_v6_02, GPU1, mandatory
check -> real smoke -> formal screen. No server job launched locally.
Source tar/checksum and GitHub save required; deployment/verification commands
in ../clip_bart_joint_v6_fix1/README.md.
