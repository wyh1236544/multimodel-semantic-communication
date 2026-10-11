## 2026-10-11 low-quality temporally fluctuating FSO V2

Additive fso_fluctuating_v2 follows TRFSO link-matched training concept without
claiming measured BiLSTM/hardware replication. Public V3 simulator waveforms fit
frozen five-tap widely-linear operators with stochastic correlated residual
replay; fresh-waveform/independent-noise holdout gate before neural training.
Stationary CIR Gamma components + OU pointing, 1ms coherence/.2ms business
spacing, quasi-static within .206us R=1 frame. DTAT-inspired5km/1550nm/.443dBkm
geometry; strong manual alpha/beta explicitly not Cn2=1e-13 equivalence.
Coherent AWG64GSa/s/4SPS/16GBaud/RRC.1 retained; no free CSI at TX or UTF8 bypass.

Same completed V1 M2 initialization for frozen B0 and trained B1 AWGN/B2 FSO.
Two 2000-pair x3-epoch FP32 joint fine-tunes; VAE/CLIP frozen. Equal optimizer
steps, valid task-loss masks, clean anchors and loss; erasure task loss zero,
failures retained. Fixed last step, GPU1, no new training-seed sweep. Full V3
waveform evaluation bypasses surrogate; report delivered/both-available and
all-frame/fallback/outage separately. Conditional image CIs do not cover weather
or training variation. Same slot/peak caps not exact actual energy. No digital
baseline/superiority or joint-vs-separate proof in this package.

Local NumPy/physics/source/deployment checks recorded in VERIFICATION.md;
Torch CPU, real saved checkpoints/COCO/GPU training unexecuted locally.
No server job launched. User-started launch.py pipeline gates checks, public
calibration and real smoke before two training arms and32-image validation.
Preserve previous modules, weights/splits/results. Source tar/checksum, conflict
checking installer, HANDOFF backup and atomic GitHub synchronization required.
Return v10_fso_fluctuating_v2_records.tar.gz; large weights/bank stay on server.
