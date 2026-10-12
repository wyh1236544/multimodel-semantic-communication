## 2026-10-12 whole-text/JSCC architecture screening V1

Additive `joint_jscc_arch_v1` implements the user's requested architecture revision,
preserving historical code, weights/results/splits and physical GPU1. Frozen
VAE/CLIP, deterministic posterior mean, <=75-content-token CLIP chunks up to128
total AR tokens, whole-sequence T32 learned compression and causal AR decoder
(teacher forcing TRAIN ONLY, greedy/EOS actual evaluation). No UTF8 bypass.

Matched fresh B1 text, B2 cross-token JSCC+public RX condition BUNDLE, B3 same plus
raw-wire-C auxiliary short-prefix/coarse-latent tasks, B4 block-diagonal neural
separate control. B3/B4 have identical registered/trainable parameter scale and
source/updates/payload; attention masks/memory lengths differ, NOT equal FLOPs.
B0 is a frozen B2 rerun reference with77 source tokens, not an AR-only causal
ablation or reproduction of historical sampled-VAE scores. All new arms share
same new initial tensors and total text capacity. Default1000pairs x(2clean+
3FSOepochs), FP32 batch1/accum2, fixed last step,32 validation pairs, one weather
draw, no training-seed sweep/final test. Clean-stage snapshots reported separately.

R=1, C32/I224/T32 width16,2304 complex payload, AWG64GSa/s/4SPS/16GBaud/RRC.1.
Source masks stay at TX; RX only observed symbols/pilot confidence/public layout,
availability/observed amplitude. No true SNR/CFO/weather/source-text RX input.
Current learned payloads require fresh full-waveform holdout fidelity gates at
initial/clean/FSO stages; reused surrogate is approximate, not hardware-trained.
Final evaluation uses full V3 waveforms, retains failures/all energy/time events.
Common task reads physical C before private RX mixing; full frame resources
charged, NOT a demonstrated low-bandwidth C-only transmission. Same slot/peak
caps do not guarantee equal actual energy. Covered-text and full-source WER
separate, truncation documented. No analog BER or promised paper-level gains.

Local NumPy/static/source/deployment checks are recorded in VERIFICATION.md;
Torch toy tests/pretrained/GPU1/COCO/AR/fidelity/performance UNEXECUTED locally.
No server training launched here. User-started pipeline gates required CPU tests
and8-pair real smoke before main run. Default4h is a stop budget, NOT ETA; AR
inference is slower, measured smoke and first100-200steps refine estimates.
Source tar/checksum, additive conflict-checking installer, HANDOFF backup and
GitHub save required. Return v10_joint_jscc_arch_v1_records.tar.gz; keep weights
on server. See ../joint_jscc_arch_v1/README.md for commands and scope boundaries.
