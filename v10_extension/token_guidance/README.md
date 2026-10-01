# Joint token guidance quick V1

2026-10-01. Additive **direction screen**, not a new proven best model. Everything
is under the existing multimodel project; old adaptive, V10, physical-link and
UTF8 business experiments, weights and records stay untouched.

## Question and actual changes

Does explicit text-guided image encoding and received-token-guided image decoding
help after equal short fine-tuning? All arms still have the original implicit
joint image/text Transformer. A is NOT image-only. The two existing modules are
controlled using process-local hooks (no old source edits or state-name changes):

| Arm | Explicit sender text→image | Explicit receiver C/T→image |
|---|---|---|
| A | off | off |
| B | on | off |
| C | off | on |
| D | on | on |
| OLD | off | on; original weights, no fine-tuning |

A/B/C/D each start from the exact same existing mixed seed42 checkpoint and
receive 512 fixed train2017 image/caption pairs ×2 epochs =1024 single-image
forwards, 512 optimizer updates at accumulation2. Only one fixed training seed
is used for this quick screen. Frozen: VAE, CLIP/projection, joint encoder, old
Common auxiliary heads, channel adapter and other legacy adaptive branches.
Train: bottleneck, joint decoder, image head, text head/direct adapter, and the
enabled explicit guidance modules. Dropout is disabled equally in all arms but
autograd is enabled. Core LR1e-5, guidance LR5e-5, AdamW decay1e-5, gradient norm1,
FP32/no AMP. The last fixed-budget checkpoint is used, with no validation
checkpoint selection or hidden hyperparameter search.

The image loss is 1.5×latent MSE +0.75×pixel MSE +0.35×(1−SSIM), plus0.5×masked
source-text CE. This is deliberately not the old Common-probe/replay objective.
No claim of pure unimodal `common_image` independence is valid when early
guidance is enabled. Cross-attention itself is an existing mechanism, not a
newness claim. Semantic source captions remain CLIP77-truncated; neural token
transmission does NOT guarantee full UTF8 word-faithful recovery.

## Communication contract

Image and text jointly produce C32/I224/T32, width16:4608 real values. No raw
UTF8 packet or original caption is available to the receiver. The complete
semantic object uses the existing8-bit [-4,4] scalar quantizer, object CRC,
512-byte packets, CRC32, K7(171,133) nominal1/2 convolutional FEC, coded-bit
repetition2, Gray16QAM, RRC rolloff0.1/span32,8-bit DAC,64GSa/s per I/Q and4sps.
Rs16GBd, gross64Gbps, nominal FEC/repetition net16Gbps before packet overhead.
Single-polarization coherent FSO field and256-symbol public preamble; one common
slow field for probe and all feature packets, no ARQ or free feedback. Fixed
reconstruction mode is public. Equal-power digital protection, NOT analog UEP.
BER, CRC failures, actual samples/scheduled times and constellations are retained.

**Training actually runs this same exact waveform/FEC/CRC forward.** Backward
uses an identity straight-through approximation through successfully decoded
quantized objects; CRC-erased objects give zero sender gradient. Viterbi/CRC are
not claimed differentiable. Receiver heads can still learn on erased samples.
Training conditions cycle AWGN20/30dB and moderate FSO20/30dB; low-SNR and strong
FSO are validation stress conditions. Forward CRC/delivery rates must be checked
before interpreting weak gradient/no gain as rejection of the idea. Whole-object
erasure matches existing V3; partial/group recovery is NOT silently introduced.

There is also a **quantized-clean** reference (ideal delivery, reliability1):
source quantization remains, but no physical waveform/noise. Its40 routing value
is an explicitly ideal decoder-gate input, NOT a measured optical SNR. It is not
included in physical rate/BER claims or primary physical gain. Physical SNR is
the existing nominal Es/N0 reference, not a hardware-calibrated receive SNR.
No laser/clock/MZM response or measured optical link-budget addition is claimed.

## Install first, then run

Upload `v10_token_guidance_quick_v1.tar.gz` and `.sha256` into existing
`/data1/zw/WYH/multimodel`. Archive contains only a new staging folder, no data or
weights. The additive installer verifies old prerequisites and refuses conflicts.

```bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_token_guidance_quick_v1.tar.gz.sha256
test ! -e _v10_token_guidance_quick_v1 && tar --keep-old-files -xzf v10_token_guidance_quick_v1.tar.gz
/home/zw/anaconda3/envs/torch/bin/python -B _v10_token_guidance_quick_v1/install.py --project-root .
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
export CUDA_VISIBLE_DEVICES=1
"$PYTHON_BIN" -B v10_extension/token_guidance/launch.py check
"$PYTHON_BIN" -B v10_extension/token_guidance/launch.py smoke experiment_results/v10_token_guidance_smoke_01
```

Smoke:four arms×8 images×1epoch, two validation images,32 evaluation rows;
30-minute cap. It tests real gradient/checkpoint/data paths, not effectiveness.
After successful completion, `timing.json` gives a rough measured quick ETA.
New GPU1 run; do not stop other users' processes or upgrade the environment.

```bash
"$PYTHON_BIN" -B v10_extension/token_guidance/launch.py quick experiment_results/v10_token_guidance_quick_01
```

Quick:four1024-forward trainings,32 fixed validation images,1792 validation
forwards, overall3-hour cap. Expected initial range **45–90minutes training;
15–30minutes evaluation; approximately1–2hours total**, not a measurement.
Old full runs took2.3–2.4hours each, but the new waveform/Viterbi CPU cost makes
direct neural-step scaling unreliable; prefer the smoke estimate. CPU contention
and GPU sharing can exceed it. All writes/logs/caches stay in multimodel. The
launcher waits and saves `output.log`; use a second terminal to see progress:

```bash
tail -f experiment_results/v10_token_guidance_quick_01.log
nvidia-smi -i 1
```

No automatic retry/resume. Interrupted runs are retained; use a new attempt name.
Four `train_*/last_delta.pth` files contain only changed parameters plus base
checkpoint and split SHA. They are NOT standalone original V10 checkpoints.
The exact old base weights are required to load/reconstruct them; old entry
points must not be pointed at a delta checkpoint.

## Validation, diagnostics and decision

All five weight conditions are evaluated on the same32 validation IDs, fixed
source captions, fields/noise, token count and packet/symbol budget. Physical
FSO moderate/strong ×0/10/20dB plus quantized-clean. Save the first4 images under
EVERY condition, not best-case selection; every row contains neural received text.

D also runs three dependency diagnostics, NOT separately trained ablations:

- `tx_wrong`: another image's caption replaces ONLY explicit early guidance;
  the original source-text joint branch and scoring target remain unchanged.
- `rx_wrong`: another image's **received/decoded** C/T memory replaces ONLY
  the explicit RX refiner memory; no original donor caption is supplied to RX.
  CRC-failed donor returns the unrefined image; both-delivered subset is reported.
- `rx_zero`: bypass only the explicit RX refiner, not the full joint text branch.

Primary screen: moderate/strong physical20dB averaged within image. Report
image-paired bootstrap4000 replicates, sign-flip4000 Monte Carlo tests and Holm
correction for six predeclared contrasts; also every condition and both-accepted
subset. `promising_expand_validation` requires D−A>=0.2dB, image CI lower>0,
Holm p<0.05 and positive clean gain. This threshold is an operational screen,
not a publishability test. Inspect SSIM/text regression, D−OLD and wrong-text
diagnostics as well before final direction. A single short seed cannot quantify
training variance; CLIP-related/text metrics are not independent object-faithfulness
proof. Need larger held-out evaluation and independent semantic/visual checks
if promising. No test images are used; a negative quick screen is inconclusive,
not proof no text-guided method can work.

Completion checks full matrix, finite scores, enabled-guide nonzero gradients
and parameter updates, equal resources/field/noise,
no raw UTF8 and four full training budgets. It does NOT require PSNR improvement.
All failures and negative outcomes remain visible. Return only small records:

```bash
"$PYTHON_BIN" -B v10_extension/token_guidance/collect.py experiment_results/v10_token_guidance_quick_01 v10_token_guidance_quick_records.tar.gz
```

Send the records tar.gz. It includes curves, timing, complete text/image/BER
records, figures and logs, excludes checkpoints/data. If smoke/check fails,
send the full error and the new log/run_identity instead of launching quick.

## Verification status

Local NumPy/source/install checks are separate from server Torch/checkpoint/GPU
tests. Read HANDOFF.md for actual performed verification. No real effectiveness
result or server run is claimed by this source release.
