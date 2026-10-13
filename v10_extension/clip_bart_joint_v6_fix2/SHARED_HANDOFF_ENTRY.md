## 2026-10-13 — CLIP/BART joint V6 fix2: independent TRAIN caption identity

User-approved follow-up after fix1 real GPU smoke passed SNR checks but finish
rejected Adapter TRAIN teacher differs from declared source. V2 joint cache's
source_sha256 hashes image float32 bytes PLUS original caption; previous verifier
incorrectly compared caption-only SHA to that joint SHA. Old synthetic fixture
used text-only digest as fake joint identity and hid the production mismatch.

Keep all original joint source hashes and normalized original captions unchanged.
V6 attach_text_source_hashes adds separate text_source_sha256 to image receipts,
cross-checking earlier independent BART caption-view hashes, fixed image IDs,
selected annotation IDs/file SHA/split/UTF8 length. Runs immediately after real
TRAIN/validation cache and before joint training. Audit metadata only, never
receiver/source tensor/length side channel. Generation teacher verifier uses
text-only digest AND independent BART TRAIN receipt, fixed IDs/split, full vector
counts; no silent short-vector zip or acceptance of joint hash as text hash.

Only V6 core/run/report/test_core/test_report/MANIFEST replacement. Model,
receiver_cache NPZ producer/schema, training/channel/config/budget/quality gates
and fix1 SNR correction unchanged. Tests execute original production joint-hash
expression using a NumPy interface shim, not a claimed Torch/GPU cache. New
realistic fixture reproduced original verifier error before repair. Local18core
and43report pass;1 actual Torch case skipped, still required server;49 historical
V5/V4 tests and8 deployment contracts run separately in release verification.
This validates metadata, NOT recovery gains or real saved-backbone execution.

Fix2 requires verified V6+fix1 installed; exact six source backups and HANDOFF,
all378 old dependencies checked, unknown/mixed edits refused, idempotent. Keep
immutable V6/fix1 archives and failed_01/_02 records unchanged. New_03 executes
checks -> real smoke -> own-gated formal screen, GPU1; no server job launched
locally, no failed-run reseal or checkpoint reuse. Source tar/checksum and atomic
GitHub save required. See ../clip_bart_joint_v6_fix2/README.md.
