# Verification boundary

Local host: bundled Python3.13/NumPy2.3.5; NO PyTorch, pretrained weights, COCO or school GPU access. No environment installation/upgrade and no server training were performed.

Executed local suites:

- `test_core.py`:5 tests — fixed symbol budget, long-text CLIP chunk limits/EOS/PAD, bounded configuration, exact training pairing, image-cluster bootstrap.
- `test_report.py`:5 tests — no hidden repeats/text oracle/fake analog BER; failures retained; complete SYNTHETIC matrix/receipt/parameter/fidelity/result-ledger checks; missing rows, altered parameter counts, training on holdout and false fidelity passes rejected; artifact tampering rejected.
- Python3.8 grammar parsing and launcher help/import without Torch.
- Unmodified predecessor regression suites: optical_realism_v3/test_dsp.py (7), test_repairs.py (9), fso_fluctuating_v2/test_core.py (7), full_unfreeze_v1/test_core.py (5). Together with the10 new tests:38 local non-Torch tests passed.

Synthetic report values are ONLY test fixtures, never real recovery scores. No GPU performance claim arises from these checks.

Release builder additionally verifies source-only USTAR members, exact restore bytes, additive install/reinstall, prerequisite preservation, old-HANDOFF backup, conflicting source and changed dependency rejection. Its actual outcome/checksum is in `releases/v10_joint_jscc_arch_v1_build_verification.json` once the builder completes.

Required server tests (`test_model.py`, NOT EXECUTED LOCALLY):6 Torch toy tests covering causal autoregressive masks/EOS-PAD targets, source positions beyond77 affecting code, equal joint/separate parameter scale and blocked cross-modal paths, Common independence from private receive slots, real parameter changes through new source/spatial/TX/RX/conditioning/AR paths and exact checkpoint reload. Missing Torch is an ERROR, not a skipped pass.

Required real server smoke: exact base+A+B2 loading, frozen backbones, current learned-payload waveform-vs-surrogate gates, actual greedy captions, real optimizer updates/checkpoint roundtrip, memory reserve and independent full result verification. Main run remains gated by this smoke. None are substituted with fabricated tensors/metrics.

Integration/paper limits: new-text capacity128 vs legacy77, fixed deterministic VAE input vs historical sampling, codec+conditioning bundle rather than single-component attribution; separate attention/memory computation differs despite same parameter count; same peak/time caps not equal actual energy; full-frame C diagnostic not a separately transmitted low-bandwidth Common payload; one training replicate/selected weather paths/historical validation; no digital baseline, independent final test, hardware measurement or guaranteed benefit.
