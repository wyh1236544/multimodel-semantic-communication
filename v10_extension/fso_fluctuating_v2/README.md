# Low-quality fluctuating FSO V2

Status: additive implementation, simulator-calibrated; real checkpoint/Torch/GPU training must pass server smoke. Never replace old modules/weights/results. Physical GPU1; no environment changes, no final test, no seed sweep. Release `v10_fso_fluctuating_v2.tar.gz` contains source/config/tests/docs only.

## What is different from V1?

V1 trained a differentiable symbol channel with independently sampled per-frame Gamma-Gamma/pointing gains. It did jointly fine-tune the semantic encoder and decoder (VAE/CLIP frozen); it was not full DSP-aware waveform training. V3 physical imperfections and rejection entered evaluation only. In the expanded validation, most all-frame mean PSNR gain came from unavailable-frame fallback, not a reliable improvement in delivered information.

This module adds time-correlated FSO amplitudes and a simulator-calibrated **stochastic** five-tap widely-linear surrogate. It does NOT pretend NumPy physical propagation is differentiable or insert an identity straight-through channel. Fit public non-business symbol waveforms through the full V3 DAC/IQ-MZM/EDFA/FSO/ASE/shot/thermal/ADC/timing/CFO/phase receiver; preserve the correlated residual as random replay rather than replacing noise by its mean. Gradients pass through the identified complex linear operator, with its parameters frozen. Payload-dependent nonlinearities are only approximately represented; PAPR shaping penalty remains differentiable. Validate on fresh public waveforms/noise before training. Afterwards evaluate image/text through the actual full waveform simulator, bypassing the surrogate.

Inspired by [TRFSO](https://arxiv.org/pdf/2509.03835): AWGN initialization, link-matched channel layer, joint transmitter/receiver optimization. That paper learned a BiLSTM from optical-bench data using IM/DD at 30GBaud. Here we have no measured data; use an explicit stochastic identified operator and retain the project's coherent 16GBaud link. This is not a reproduction of that hardware or its MS-SSIM gain.

Geometry follows selected assumed [DTAT-FSO](https://arxiv.org/pdf/2409.11928) parameters: 5km, 1550nm, .443dB/km, .25mrad FULL divergence, .2m receive aperture, 1ms turbulence coherence. Gaussian beam waist/receiver coupling/EDFA/noise/ENOB parameters remain project's declared assumptions. Weak/moderate Cn2=1e-17/1e-15 produce plane-wave Gamma-Gamma alpha/beta. Strong is **manual alpha=2.1,beta=1.2**, NOT a claimed equivalence to Cn2=1e-13 at 5km: that exceeds the supported plane-wave domain. No phase-screen propagation, weather measurements, complete OptiSystem equivalence or calibrated sensitivity claim.

## Temporal/channel contract

- Exact stationary CIR updates generate the two Gamma factors; Gaussian OU x/y pointing gives the circular-aperture collection integral. Components' lag correlation exp(-.2ms/1ms)=.8187; the product's correlation is not exactly .8187.
- Each ~.206us optical frame has constant atmosphere/pointing. Successive BUSINESS frames are spaced .2ms; this models a sampled weather trajectory, NOT transmitting the same image again or a .2ms optical airtime. Frame duty and idle time are separate from waveform duration.
- Same weather path across methods AND launch-power points within a weather class. Independent train, public holdout and evaluation streams; each evaluation draw is a distinct trial, not combined repetition. Carrier phase resets per independent business frame; continuous phase/clock state within its probe+payload is retained. Not a model of one uninterrupted long optical capture across all business frames.
- SNR is derived from power/noise; not set to a desired dB after fading. Public pilot estimate, offline optical/shot/thermal noise-only projection SNR, EVM and receiver lock are different quantities. Converter/nonlinear distortion is included in EVM, not noise-only SNR.
- AWG64GSa/s,4SPS,16GBaud,RRC.1/span32; C32/I224/T32,width16,2304 complex payload symbols; R=1; charged probe/tracking pilots/filter tails/guard:13178 samples, .20590625us. 4us slot is a ceiling, not spent airtime. Single polarization. No new UTF8 side channel, no fictitious analog BER.
- Same fixed public laser workpoint/peak envelope/slot cap per condition; actual energy is recorded and may differ. This is not strict equal-energy continuous-versus-digital superiority.

## Models/budget

All three start from the EXACT completed V1 `M2_awgn_ft` checkpoint, not selected best by new results. Frozen B0; B1 AWGN fine tuning; B2 calibrated fluctuating-FSO fine tuning. Same 2000 training pairs ×3 epochs, batch1/accum2=3000 optimizer steps per trained model. Both train eligible semantic encoder/bottleneck/decoder/heads/TX-RX maps; VAE/CLIP frozen and checked unchanged. FP32, no AMP. Same ordering, public channel schedule, task losses and PAPR penalty. B1 uses calibration pilot-SNR schedule clipped to [-20,30]dB (logged), with fresh AWGN; B2 uses calibrated linear response plus stochastic residual. No SNR input or free CSI at TX.

15% common clean anchors REPLACE steps, not add free training. Both arms use SAME unavailable-frame task-loss mask, so B1 does not get more valid supervised pairs; unavailable-frame image/text task gradient is zero, PAPR still optimized. No receiver-prior training on erasures to inflate fallback PSNR. Failed frames remain in histories/evaluation. Fixed last-step checkpoint, no test/validation selection.

Public calibration: 48 training +12 holdout frames per six conditions, plus fresh-waveform holdout checks. Bank gate requires >=2 jointly available holdout frames per condition, relative mean EVM error <=.5, acceptance disagreement <=.25, some valid training frames. These are permissive, predeclared engineering checks, not full distribution fidelity/hardware certification. Failure stops before training; do not silently relax thresholds or retry until passing. Residual tails, content-dependent lock and nonlinear mismatch remain limitations.

Validation:32 fixed validation-prefix pairs ×(clean +6conditions×2independent draws)×3models=1248 rows. 1152 full-waveform rows;96 clean controls. These IDs may have been used historically: not independent final test.

## Safe install, existing multimodel only

Download BOTH source archive and `.sha256`, upload with your existing SFTP connection to `/data1/zw/WYH/multimodel/`. Then:

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_fso_fluctuating_v2.tar.gz.sha256
test ! -e _v10_fso_fluctuating_v2 && tar -xzf v10_fso_fluctuating_v2.tar.gz
"$PYTHON_BIN" -B _v10_fso_fluctuating_v2/install.py --project-root .
"$PYTHON_BIN" -B _v10_fso_fluctuating_v2/install.py --project-root . --verify-only
```

Installer checks source hashes and prerequisites before writing; adds ONLY this module, backs up HANDOFF before appending. Identical reinstall accepted; different old source/dependencies refused. Never overwrite to get past a conflict. Extraction stage is immutable; no Mac extended headers. Need existing fso_jscc_v1, fso_jscc_eval_v1 and V3 modules plus the original server weight/data directories. This tar is additive, not a whole-project replacement.

## Run (user starts)

Check ownership/available GPU memory; do not stop others' processes:

```bash
nvidia-smi -i 1 --query-gpu=index,name,memory.used,memory.free --format=csv
"$PYTHON_BIN" -B v10_extension/fso_fluctuating_v2/launch.py check
```

Recommended single pipeline: CPU contracts → public simulator calibration (30min cap) → real saved-checkpoint 8-pair smoke (15min cap) → TWO trained arms → waveform validation → independently verified report/archive.

```bash
"$PYTHON_BIN" -B -u v10_extension/fso_fluctuating_v2/launch.py pipeline \
  experiment_results/v10_fso_fluctuating_v2_01 --max-hours 12
```

Defaults: base `experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth`; A `experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth`; source `experiment_results/v10_fso_jscc_v1_01/v10_screen`. To inspect only the channel without neural training:

```bash
"$PYTHON_BIN" -B -u v10_extension/fso_fluctuating_v2/launch.py calibrate \
  experiment_results/v10_fso_fluctuating_v2_channel_01 --max-hours 1
```

The pipeline forces physical GPU1 (program sees cuda:0). New output/log required; choose `_02` on an existing path, no deletion/resume. Budget expiration preserves partial records and no COMPLETE. Only owned child PID may be terminated on timeout. Empty PYTHON_BIN falls back inside launcher, but shell invocation still needs the explicit export above.

Progress, from another terminal:

```bash
cd /data1/zw/WYH/multimodel
tail -n 20 experiment_results/v10_fso_fluctuating_v2_01/calibration.log
tail -n 20 experiment_results/v10_fso_fluctuating_v2_01/v10_screen.log
```

Training logs show measured projected per-arm time every20 forwards; first100–200 steps refine ETA. Historical V1 took15–16min/4000 forwards per trained arm; this version12000 total forwards suggests ~45–50min training IF throughput stays similar, plus load/cache/smoke/physical evaluation; tentatively1–2h total, NOT a guarantee. 12h overall/4h per arm are stop limits, not predicted duration.

## Return/results

Return `experiment_results/v10_fso_fluctuating_v2_01/v10_fso_fluctuating_v2_records.tar.gz`. Contains calibration summary, histories, complete identities/hashes, summary, full captions, images/constellations, report.html and quality_timeline.svg; excludes large weights and bank.npz (keep on server). Failed pipeline: return failing log and calibration/bank.json, not a fabricated complete archive.

Primary: B2−B1 detailed-waveform PSNR on configured low-quality conditions, **both receivers available**. Also all-frame PSNR, conditional SSIM/text quality, outage, fallback/discordant contribution, p10 quality, actual energy and time. Image-bootstrap intervals are conditional on two fixed weather trajectories and one trained replicate; they do not quantify training/weather uncertainty. A positive conditioned gain alone is not a causal goodput claim because availability is post-treatment. If only fallback improves, no delivered-semantic gain. No claim of continuous>strong digital, joint>separate, compression or hardware equivalence; those require separate fair experiments. Clean text ceiling still limits original-caption fidelity.
