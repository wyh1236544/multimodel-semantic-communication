## 2026-10-12 text-only JSCC frozen-decoder initialization repair Fix1

User's server evidence: original8 Torch toy tests passed, but real GPU smoke
failed native_overfit learning audit with46.09GiB free. Frozen old model
initialization exposed an implementation defect: ARTextDecoder deep-copied
old.decoder and retained requires_grad=False. Original toy fixture did not
freeze its reference, so could not cover that production path.

Additive text_jscc_v1_fix1 keeps text_jscc_v1 and failed outputs unchanged.
Only the independent candidate language copy is explicitly trainable; original
CLIP/B4/VAE remain frozen/unaliased and their hashes/gradient checks remain.
All text-frame/model/loss/data/schedule/quality/fidelity thresholds stay the
same. No removed audit, longer-training workaround or expanded unfreeze.

Training now writes per-prefix trainability before optimization and refuses
frozen required candidate groups early. learning_audit.json records exact
gradient/update failures before raising. Torch toy fixture now uses frozen
references in all8 existing tests; adds2 explicit frozen-copy trainability/
no-alias/old-weight preservation and real decoder-gradient/update regressions.
Server must rerun all10 Torch tests and real short smoke. Local31 non-Torch
contracts/source/deployment checks are separate; Torch/real repair effectiveness
UNEXECUTED locally. No server training launched here.

81>77 tokenizer warning is not this RuntimeError: raw tokenization may exceed
CLIP's77 capacity but CLIP forward already uses<=75content+BOS/EOS chunks.
Do not silently truncate original texts or enlarge CLIP positional embedding
to hide the warning.128-token candidate coverage contract remains unchanged.

Same GPU1/FP32/384complex/R=1, fresh output
experiment_results/v10_text_jscc_v1_fix1_01; old _01 retained. New additive source
tar/checksum and GitHub save, HANDOFF byte-backup then append required.
Return v10_text_jscc_v1_fix1_records.tar.gz after completion/quality stop; on
execution failure return smoke log, FAILED and available training audit.
See ../text_jscc_v1_fix1/README.md. No recovery-improvement/hardware claim.
