# VAE tail ablation V1 verification — 2026-10-12

Executed locally: bundled Python3.12.14/NumPy, no Torch/pretrained weights/COCO/server connection.

- `test_core.py`:5 tests passed, exit0. Exact terminal decoder prefixes, unchanged channel, bounded1000×3 design,2496row matrix, preselected image cohorts, invalid configs, completed-V2/base/A/split/calibration identity and paired schedule negative tests.
- `test_report.py`:1 integration test passed, exit0. Real NumPy V3 events with explicitly SYNTHETIC neural scores:42row full matrix+6diagnostics, parameter/frozen/tail-change audit, HTML/timeline report; mismatched latent/frozen-tail audit/duplicate samples refused. Not real model performance.
- Existing V3 `test_dsp.py`:7 tests passed; `test_repairs.py`:9 passed, exit0. Total22 local numerical/contract tests.
- Python3.8 grammar parse: passed. Torch availability checked: unavailable. NO claim of local Torch backprop/real checkpoint loading or GPU gain.

`build_release.py` additionally checks immutable source archive, extraction/restoration, exact prerequisites, additive install/verify/reinstall, no-overwrite conflict refusal, changed prerequisite refusal before writes and original HANDOFF backup. Its emitted `releases/v10_vae_tail_ablation_v1_build_verification.json` is the actual build evidence; archive creation alone is not numerical verification.

NOT executed locally: new Torch CPU toy-gradient/update/replay tests; real AutoencoderKL tail-name/gradient check; exact saved V2 B2/base/A loading; GPU1/COCO training; actual PSNR/text/latency/memory gains; hardware measurements. Server launcher requires both Torch CPU tests and a genuine8-pair smoke before screen. User initiates server runs; no job launched in this development task, no dependencies installed or upgraded.

Original V2 source/configs and all historical weights/results remain unchanged. New module plus backed-up HANDOFF append only. Atomic authenticated GitHub save on inspected main with expected-head lease is required; no force push. Scientific/compute limits in README.md.
