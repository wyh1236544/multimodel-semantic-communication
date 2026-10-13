## 2026-10-13 — staged CLIP/BART joint V6 and separate generation diagnostics

User authorized implementation after V5 formal review: joint-separate mean
PSNR+0.00584dB, J1/J2 captions224/224 identical, common fusion hurt cleanPSNR
by0.125dB. Preserve V5/fix1, weights/data/splits/results and existing environment.
New clip_bart_joint_v6 only, physicalGPU1, source-only archive/GitHub save.

Verify actual V5_02 formal screen and its own P0/3000 text checkpoint hash.
Continue TEXT weights only in two3000-update blocks, record3000/6000/9000 full
validation controls and native-source-memory diagnostic explicitly NOT transmitted.
Optimizer/LR/data schedule reset perblock; no precise resume or validation-best
selection. Fixed9000 last text gate before joint. No TRAIN8 or smoke initialization.

Balanced image16/text16 source pooling to C32x16; private I224x16 and T32x64,
same3072complex/full resource budget/R1. Same three registered-size/init/data/
order/loss/resource arms J0 separate SAME frame, J1 joint, J2 noCLIP TEXT SOURCE.
Frozen BART/CLIP/VAE/B4, Common2epochs independently caption/coarse/alignment
warmup with private codecs/normal fusion frozen. Joint8epochs adds25% private
receiver-token dropout on a4-step schedule with full anchors, auxiliary caption
loss, zero-init bounded caption-memory and4x4 spatial-coarse residual routes.
Dropout is synthetic TRAIN receiver-feature damage, NOT FSO loss/bandwidth gain.
All arms matched. Common-only caption diagnostics, heldout masked validation
normal/Czero/fusionoff/Cshuffle, plus ordinary full waveform eval/failed frames.

Default screen clean only. Optional pipeline reruns stages then own-gated paired
clean_continue/FSO proxy forks from SAME clean weights, not exact continuation of
screen directory. Existing surrogate/full-waveform fidelity boundaries unchanged.

Separate generate.py: existing complete local StableDiffusionImg2Img (not VAE
only), frozen diffusion/text/VAE, small train-only C+pilotSNR -> prompt residual
adapter with SD text-embedding teacher. NOT DiffJSCC/ControlNet/DiT diffusion
training/replication. Teacher/source text TRAIN only; eval prompts actual RX.
Fixed first4 evalIDs/allconditions: direct/image-only/RXtext/RXtext+CSIzeroC/
RXtext+C+CSI/shuffledC; shared seeds/strength/steps, reject failed receivers,
no candidate selection. SD weight path unknown, no downloads/upgrades. PNG
PSNR/SSIM measured; LPIPS/FID explicitly unavailable/unvalidated, no perceptual
success claim. Receiver NPZ schema forbids source labels; TRAIN cache disjoint.

Local42 new nonTorch contracts + existing regressions and actual release restore/
install checks recorded in README/verification.26 Torch/HF toy tests, saved
models/GPU/SD/actual learning NOT executed locally (Torch unavailable). Automatic
server checks+smoke mandatory. Main16GiB startup/4GiB reserve, disk12GiB,4h cap;
rough60-150min screen UNMEASURED, refine live logs. No task started on server.
First command launch.py screen NEW v10_clip_bart_joint_v6_01. Return lightweight
records even quality stops. See ../clip_bart_joint_v6/README.md.
