## 2026-10-12 fixed-B4 receiver residual fusion V2

Additive residual_fusion_v2 implements the authorized next diagnostic screen,
not a validated fix. Source is EXACT completed joint_jscc_arch_v1 B4 separate
fso_delta plus exact original base/A/B2/bank identities. All B4, CLIP/VAE and
historical parameters stay frozen, hashed and gradient-free. No new pretrained
model download, global/Conda change, unrelated process action or final test.

First, two fresh AR decoder TRAIN-only overfits (32 samples,1000updates each):
D0 precompression source-feature oracle diagnosis, D1 clean T32 received memory;
D2 evaluates the SAME D1 decoder through full FSO, no noisy retraining. Actual
greedy generation, EOS/cap termination and covered WER audited independently.
Formal D0/D1 WER<=.10 engineering gate; failure yields DIAGNOSTIC_COMPLETE and
diagnostic_stop archive, NEVER main COMPLETE. Small8-update smoke bypass is
explicit and never records a formal quality pass. Overfit weights are diagnostic
only and are NOT used as production text decoders.

If gate passes: frozen B4 reference plus two matched49,505-parameter residual
arms, R0 image-side vs R1 received-Ctext/T side. Zero output initialization,
availability gate, trainable attention/gate, clean->FSO training with paired
drop/shuffled-donor augmentation and actual gradient/update/reload checks.
No Common auxiliary task or transmitter-joint coding change in this stage.
Default1000pairs x(2clean+2FSO),4000forwards/2000updates per arm. Same fixed last
step,FP32,batch1/accum2,GPU1; startup16GiB/free4GiB reserve,3h overall stop cap.

Final full-V3 waveform runs ONCE per image/condition/draw. Frozen/R0/R1 and
off/zero/shuffled interventions reuse EXACT received tensors/events, so actual
energy/time/noise are identical within this decoder-only comparison, not just
equal peak caps. Analog R=1,C32/I224/T32 width16,2304complex,UTF8 bypass0. AWG
64GSa/s,4SPS,16GBaud,RRC.1 and physical parameters unchanged. Current fixed B4
payload fresh waveform/holdout fidelity checked before surrogate training.
All failures,actual optical power/SNR/energy/timing,images,text and constellations
retained. Main text captions are the same UNCHANGED B4 decoder for every arm.

Default32 validation pairs x7conditions x6 decoder variants=1344rows;384 actual
probe/features events, not 6x uncharged/free physical retransmissions. Conditional
image bootstrap; single trained replicate/fixed weather/historical validation.
Not proof source-joint coding beats separate, optimal image-only resource use,
continuous>digital, compression, hardware/paper reproduction or publication.

Local non-Torch and source/deployment checks in VERIFICATION.md. Torch/pretrained
B4/GPU/real EOS/gradient/performance UNEXECUTED locally; no server job launched.
User pipeline gates real Torch contracts and8-pair smoke before formal diagnosis.
Return v10_residual_fusion_v2_records.tar.gz (also when diagnostic_stop); keep
large weights/bank on server. Source tar/checksum, recoverable HANDOFF append
and GitHub save required. See ../residual_fusion_v2/README.md.
