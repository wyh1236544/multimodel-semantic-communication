## 2026-10-12 paired partial-VAE-decoder ablation V1

Additive `vae_tail_ablation_v1` keeps the multimodal model and the SAME completed
V2 B2 FSO initialization/calibration for U0 frozen, U1 FSO frozen-VAE fine tuning,
and U2 same plus last VAE decoder up block/output layers. CLIP/VAE encoder/early
decoder frozen and checked unchanged. Both trained arms1000pairs×3epochs, FP32,
same steps/data/channel/network random draws/task masks/loss/LRs; tail LR1e-6.
Physical GPU1, no environment changes, no seed sweep or final test. Extra trainable
parameters/compute/memory logged rather than claimed equal network training cost.

Direct-VAE and identical frozen-U0 semantic latent diagnostics separate decoder
effects from full-model recovery; not theoretical upper bounds. Full V3 R=1
waveform validation64pairs:2496rows plus192diagnostics; predeclared primaryU2-U1
on positions32-63, four original low-quality conditions, both available; all-frame,
fallback/outage and clean/text separately. Validation cohort not guaranteed
historically unseen. Same peak/slot not strict equal actual energy. No digital
baseline or proof joint>separate/continuous>digital/paper or hardware reproduction.

Local22 NumPy/contract/regression tests and source/deployment checks documented in
VERIFICATION.md. Torch/real B2/AutoencoderKL/COCO/GPU unexecuted locally. No server
task started. User-started pipeline gates CPU tests and real8-pair smoke before
paired training; reuse completed V2 bank, no calibration rerun. Tentative35-60min
total, refine from measured first100-200steps;3h cap is not ETA.

Source tar/checksum, additive installer, original HANDOFF backup and atomic GitHub
save required. Old modules/weights/results preserved. Return
v10_vae_tail_ablation_v1_records.tar.gz; keep new deltas/bank on server.
See ../vae_tail_ablation_v1/README.md for upload/run/progress commands and limits.
