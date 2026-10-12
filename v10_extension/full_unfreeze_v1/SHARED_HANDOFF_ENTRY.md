## 2026-10-12 paired full-unfreeze screening V1

Additive `full_unfreeze_v1` preserves all historical modules, weights and results.
U0 is the same completed V2 B2 FSO model; fresh U1 terminal-VAE-tail and U2 FULL
registered-model eligibility use the same initialization,500pairs×2epochs,
loss/reference/posterior epsilon, noise/order/anchors, semantic/tail LRs and
R=1 payload resources. Full eligibility does NOT activate inactive branches:
eligible vs observed nonzero-gradient parameter counts are separately audited.
CLIP text backbone/projection and VAE encoder/decoder MUST show real gradients
AND updates in U2; training/model evaluation never bypasses the live VAE encoder
with cached input latents. Cached frozen-B2 targets are LOSS references only.

Physical GPU1,FP32,128x128,batch1,accum2,sequential arms. Startup free16GiB and
sampled4GiB reserve,64-pair32-update real-memory smoke before main1000-forward
arms; owned-child timeout/exception stops, preserves failure, no auto retry,
no other-process actions, no environment changes, no seed sweep or final test.
Completed V2 calibration reused; new full-V3 waveform validation64pairs2496rows
and192diagnostics. Noisy-minus-clean paired margin, PSNR/SSIM/text/errors/outage,
power/SNR/resources/images and constellations included. All frames retained;
image CIs conditional on fixed traces/one training replicate, not training/weather
uncertainty. Same slot/peak caps are not exact equal actual energy.

Local non-Torch/source/deployment verification in VERIFICATION.md; actual Torch
toy contracts/real saved-checkpoint/COCO/GPU full-backbone training NOT executed
locally. No server process launched here. Tentative30-90min total planning range,
refine from64-step smoke and first100-200 main steps;4h safety cap is NOT ETA.
Short screening cannot prove that sufficiently trained full models fail, nor
continuous>digital, joint>separate, paper/hardware replication or publication.

Source tar/checksum, additive installer, byte-preserved HANDOFF backup and
atomic GitHub save required. User-started full_unfreeze_v1/launch.py pipeline
in a NEW project-local output gates check/smoke/main/verification/collection.
Return v10_full_unfreeze_v1_records.tar.gz; large new parameter deltas stay
on server. See ../full_unfreeze_v1/README.md for installation/run/limits.
