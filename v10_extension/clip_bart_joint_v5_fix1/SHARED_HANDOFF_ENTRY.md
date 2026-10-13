## 2026-10-13 — CLIP/BART joint V5 fix1: secondary beam records

User-approved report-only repair after server smoke completed all three arms
but finish rejected V4 matched_beam4 rows. No CUDA OOM observed (46.90GiB free).
V4 private_controls writes matched/zero/shuffled/matched_beam4; joint controls
write only three. Prefit verifier now explicitly requires all four, recomputes
all text metrics, rejects duplicate/missing/unknown/split errors, and returns
only original three greedy controls for unchanged quality gates. Beam decode
cannot rescue a failed greedy gate. No training/network/channel/config changes.

Eight new regression tests cover actual four-control schema and negative cases;
20 report tests pass locally. No new server GPU/training verification performed.
Patch install preflights known old/new source hashes and all historical deps,
backs up report/test/manifest/HANDOFF before replacement, and is idempotent.
Original release archive and failed experiment_results/v10_clip_bart_joint_v5_01
stay unchanged. Rerun screen in NEW v10_clip_bart_joint_v5_02 (full smoke first).
Do not delete FAILED.json or reseal historical failed records as success.
Deployment and actual verification details: ../clip_bart_joint_v5_fix1/README.md.
