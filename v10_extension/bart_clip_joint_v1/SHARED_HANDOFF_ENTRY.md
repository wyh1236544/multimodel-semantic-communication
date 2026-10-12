## 2026-10-12 — additive CLIP+BART C/I/T candidate V1

User approved architecture then implementation, not dependency installs, remote
training or weights download. New bart_clip_joint_v1 only; preserve original
text_jscc_v1/fix1, B4, all source/results/weights and split. Source archive,
checksum, additive verified install and GitHub atomic source save required.

Complete offline facebook/bart-base encoder/decoder and tied pretrained LM
head, independent BART tokenizer; frozen CLIP full vision/text towers and VAE.
Train ordered128->T32x16 bridge, C32x16 image/BART/CLIP fusion and RX adapters;
reuse exact frozen B4 separate image I224 plus new TX/RX/common residual.
Full2304complex/R=1, AWG64GSa/s/4SPS/16GBaud/RRC.1 unchanged. No UTF8 bypass,
source length/mask/groundtruth at RX, or invented analog BER/CRC. Recovered
text-to-image secondary branch explicitly deferred. C/I/T not proven disjoint.

J0 removes CLIP SOURCE INPUT but retains same CLIP training teacher; J1 joint;
J2 source attention/RX task memory separated. Same registered/trainable bridge
count and public budget; not equal active parameter/FLOP/actual energy claim.
Each own clean checkpoint splits into equal-update clean_continue vs FSO.
Current-payload fresh full-waveform holdout fidelity before/after continuation;
final evaluation real V3 simulator, not proxy. No hardware/paper efficacy claim.

First run diagnostic: actual-backbone gradients + T32-only TRAIN overfit
8texts/1200forward/600optimizer steps, first200feature warmup, matched/zero/
shuffled greedy recovery. Gates WER<=.20 and memory margins>=.10. Default
screen256train/16val, 3candidates x(clean4epochs + two2epoch continuations);
672evaluation rows. Explicit diagnostic_ready/overfit_stop/clean_validation_stop
markers; only full non-smoke screen uses COMPLETE. No test/calibration business
access, validation last-step diagnostics not checkpoint selection.

Local22 non-Torch tests passed before packaging; 13 Torch/HF toy contracts,
real offline weights, actual gradients/GPU memory/effect pending SERVER.
No training launched here. Physics/install/restore tests reported separately.
PhysicalGPU1, >=16GiB startup/free>=4GiB reserve. One seed screening. First
diagnostic rough15-60min, main2-6h UNMEASURED; finite caps not ETA. Existing
Python3.8/Torch1.12/Transformers4.30.2/Diffusers0.20.2 unchanged.

Need pretrained/bart-base complete official FP32 bin/tokenizer files (expected
download aadd2ab0ae0c8268c7c9693540e9904811f36177, approximately558MB) and full
CLIP weights/preprocessor config. No automatic internet fallback. Base/A/B2/B4
exact identities required; source/legacy/calibration path overrides supported.
First command launch.py diagnostic experiment_results/v10_bart_clip_joint_v1_diagnostic_01
--max-hours2 (write with spaces as README). Send generated small
v10_bart_clip_joint_v1_records.tar.gz including actual text and learning audits;
weights remain server. On failure send current child log/FAILED/audit. See
../bart_clip_joint_v1/README.md. HANDOFF backed up before appending this entry.
