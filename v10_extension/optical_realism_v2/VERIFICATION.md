# Verification — optical realism V2

Date: 2026-10-05. Local Python 3.12.14 + NumPy/Pillow; server target remains
Python 3.8 / existing torch1.12 environment. No dependencies installed or changed.

## Executed locally

- New receiver/impairment suite: **7/7**. Arbitrary continuous (non-QAM) payload
  with +50MHz complex CFO, fractional timing, and -60/+20/+80ppm sampling clocks;
  estimated clock within3ppm and MSE<0.001 in this isolated waveform test.
  Receiver output unchanged when simulator truth fields are changed without
  changing observed samples. Noise-only/zero-signal rejection. ENOB extra noise
  variance excluding ideal quantizer variance. Exact pilot/tail sample costs.
  Over-budget repetition rejection and no low-power digital false acceptance.
- New optics/packet/codec/report suite: **24/24**. Fixed workpoint, same paired
  probe/path, passive signal-energy cap, true RRC overhead, CRC/UTF8/JPEG roundtrip,
  analog BER null, missing/duplicate/tampered rows rejected, immutable collection.
- Preserved regression suites: **32/32** optical SI, **24/24** final comparison,
  **11/11** physical waveform/FEC. Total **98 passing tests** across five suites.
- 15 B2B conditions: 3 payload alphabets × 5 impairment presets, all functional
  gates passed. Optical ASE/shot/thermal/bandwidth retained; explicit fixed30dB
  attenuator; random payloads, not neural images. EVM ranged0.1680–0.1939. These
  numbers are diagnostic simulation outputs, NOT measured hardware quality.
- Additional whole semantic-Q8 object on 16QAM+conv+repeat2: all4608 values passed
  object CRC with0 post-FEC errors in one good-link synthetic case; resource count
  matched248335 samples. A single success is not a BER reliability certification.
- New Python files parsed with Python3.8 grammar; top-level inference-runner
  imports and launch help work without needing torch at import time.

Persisted local evidence (relative to project root):

- `experiment_results/optical_realism_v2_verification_01/verification.json`
- corresponding five `.log` files in that directory
- `experiment_results/optical_realism_v2_local_b2b_01/b2b.json` and `COMPLETE.json`

The B2B CFO statistic is the initial preamble estimate; subsequent phase pilots
track its residual. With nonzero linewidth it need not equal configured LO
offset exactly. RX CFO has the opposite sign to LO-minus-TX configuration.

## Release checks

`build_release.py` creates a source-only USTAR gzip with no macOS xattrs, restores
it to an isolated project-local directory, verifies every source hash, exercises
additive install/reinstall, verifies the shared-handoff backup, rejects a changed
existing source, and rejects missing prerequisites. It writes the actual outcome
and archive SHA256 to `releases/v10_optical_realism_v2_build_verification.json`.
That generated record, not this description, is the evidence of a completed build.

## NOT executed locally

PyTorch is absent in the available local runtime. Actual torch hooks, base+A delta
loading, GPU1/COCO end-to-end image/text evaluation, server Python3.8 execution,
remote long runs, laboratory hardware, ENOB/laser calibration were **not** run.
No new PSNR/text-quality improvement is established. The server pipeline first
runs the existing `final_compare_v1/test_model.py` and a two-image smoke; it stops
on errors before proceeding to the 8-image comparisons. No server job was started.

This is a bounded candidate link/DSP upgrade. It retains block fading, assumed
coupling, zero-phase device filters, whole-object digital erasures and the frozen
neural source-quality ceiling. See README for boundaries; no claim that all
research recommendations, spatial optics, optimal digital coding or real-time
hardware latency are implemented.
