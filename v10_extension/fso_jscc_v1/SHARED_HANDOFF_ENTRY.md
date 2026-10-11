## 2026-10-11 single-shot FSO-aware continuous JSCC V1

New additive fso_jscc_v1 trains frozen-A initialization in clean_ft/AWGN_ft/FSO_ft
arms under identical optimizer steps, architecture, image order and task losses;
M0 remains frozen. VAE/CLIP remain frozen; joint semantic encoder, bottleneck,
decoder/heads and identity-initialized TX/RX residual maps are eligible. Required
nonzero gradients AND parameter updates are verified. FP32/no AMP, physical GPU1.
Fixed C32/I224/T32 width16, 2304 complex payload symbols, R=1, no source-caption
receiver side channel, no hidden repeats or fictitious analog BER.

Coherent Gamma-Gamma INTENSITY becomes sqrt(intensity) field gain with pointing,
path loss and phase. Pilot CSI only; noise fixed before fading. Differentiable
symbol proxy training is explicitly NOT full waveform/hardware training.
Independent V3 64GSa/s/4SPS/RRC.1 waveform validation uses R=1, charged pilots,
lock rejection and common slot/peak-envelope caps; actual launch energies logged,
not asserted equal. All failure samples kept; paired image bootstrap excludes
training-seed uncertainty. This does not independently prove joint-vs-separate
or analog-vs-digital superiority, compression, or hardware calibration.

Local NumPy/source/deployment tests executed; local Torch CPU and real A/COCO/GPU
training NOT executed (Torch unavailable). User-started pipeline gates short
real smoke before direction screen. Default screen: 2000 images x 2 epochs per
trained arm, 32 validation images; 128 validation can be a separately budgeted
confirmation after direction screen. No server job launched by this update.
Return v10_fso_jscc_v1_records.tar.gz; keep large new deltas on server. See
fso_jscc_v1/README.md. Historical code, checkpoints and results preserved.
