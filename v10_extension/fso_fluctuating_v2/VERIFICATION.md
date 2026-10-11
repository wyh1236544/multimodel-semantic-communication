# V2 verification and limits — 2026-10-11

Executed locally with bundled Python3.12.14/NumPy (no Torch, no COCO, no checkpoints, no server connection):

- Python3.8 grammar parsing of all new Python source: passed.
- `test_core.py`:7 tests, exit0. Gamma CIR stationary mean/variance/lag, temporal path/field identity, fixed resources, source-independent waveform operator, training/holdout separation, invalid config rejection, fallback-only gain counterexample.
- `test_report.py`:1 integration test, exit0. Real NumPy V3 link events with explicitly SYNTHETIC image/text scores:42 rows, full matrix/resource audit, report/quality-timeline generation, duplicate-row rejection. Not neural performance evidence.
- Existing V3 `test_dsp.py`:7 tests, exit0; `test_repairs.py`:9 tests, exit0. Total24 numerical/audit tests, not real-data gain.
- Public power audit with8 non-business frames per condition, before freezing low-power targets: moderate -18/-21/-24dBm yielded median offline projection noise-only SNR approximately -.69/-3.75/-6.86dB on that short path. Strong -15/-21dBm medians .86/-5.27dB. Those are pilot-audit path values, not prescribed SNR or population weather estimates. Final configured strong second point is -18dBm; path distributions vary between streams.
- Full public-waveform calibration:6conditions×(48train+12holdout)=360 identified frame operators and72 fresh independent-waveform/noise holdout transmissions. Six fidelity gates passed. Mean-EVM relative errors .02%–8.11%; acceptance disagreement0 in the paired-power trace run. Conditions' training pilot SNR p10/median/p90:

| Condition | p10 | median | p90 | train lock fraction |
|---|---:|---:|---:|---:|
| weak_p06 |13.13|13.89|14.42|100%|
| moderate_p18 |-.08|2.69|5.59|100%|
| moderate_p21 |-3.18|-.15|2.30|100%|
| moderate_p24 |-5.62|-3.16|-.34|93.75%|
| strong_p15 |-3.16|2.91|7.85|93.75%|
| strong_p18 |-5.39|.04|4.76|89.58%|

These are SIMULATOR/public-waveform records, NOT trained-model image scores. Holdout tests the approximate operator family on fresh input/noise after per-path public fitting; it does not certify a learned universal channel model on arbitrary unseen weather. Calibration partition never enters neural training. Correlated residual replay remains finite-bank approximation; nonlinear/PAPR-dependent response, content-dependent lock and unseen extremes must be checked in independent full-waveform image evaluation.

Source archive builder checks hashes, Python grammar, isolated extraction, installation, identical reinstall, conflict refusal without overwrite, wrong dependency refusal before write, and shared-HANDOFF backup. Builder writes separate archive SHA-256 and build-verification JSON. Do not infer these checks passed until the corresponding build-verification exists.

NOT executed locally: new Torch CPU parity/backprop/erasure tests, actual V1 M2/base/A loading, GPU1 training, COCO image/text recovery, waveform model performance, hardware calibration. Launcher requires server Torch tests and real saved-checkpoint smoke before the bounded screen. No school-server task launched by this development turn.

Previous local public calibration attempts remain in new project-local directories; no history deleted. New module only; old source/weights/results untouched. GitHub synchronization uses an atomic commit on the inspected parent with expected-head lease, not a force push or a local Git checkout. Deployment commands and scientific caveats are in README.md.
