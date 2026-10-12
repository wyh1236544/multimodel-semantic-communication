## 2026-10-13 — additive decodable BART latent codec V4

User approved literature-backed implementation after failed Text V3 diagnosis.
New bart_latent_codec_v4 only; old modules/C/I/T/data/splits/weights/results
preserved. Frozen existing offline BART; CLIP/VAE/images not loaded. Independent
LD4LG-inspired learned32 queries and fixed32x768 reconstruction memory; no
four-position folding/128-position expansion or original-position MSE. No new
diffusion model, LoRA, backbone unfreeze, dependency install or remote training.

Paired P0 CE/P1 CE+word-distribution KD share exact initial tensors/data/order/
updates/payload. Teacher gold-prefix original BART features TRAIN only, not RX.
Default32x64=1024complex; optional32x32=512/32x16=256 in independent new runs.
Per-slot unit power,R1,no UTF8 side channel/free source mask/length. RX public
all32 mask, output EOS; beam4+repeat restrictions secondary, unrestricted greedy
primary. Native source-memory TRAIN oracle and actual incremental/cache check.

Default two TRAIN8 diagnoses,1200optimizer updates each,batch4,FP32,one seed42.
Same WER.20/memory margins.10/naturalEOS gate; failures retained, no relaxation.
Optional own-passing arms reset UNTRAINED initial codec for1024train/3000updates,
validation32 WER.60/same margins/EOS before any FSO. Same clean checkpoint forks
clean_continue/FSO_continue,1000updates each; same frozen clean-memory teacher
and consistency/anchor recipe. Gamma-Gamma/pointing/pilot-estimated symbol proxy
explicitly omits temporal fading/waveform/DAC/MZM/sync; nominal SNR not derived
optical power. Full fresh compact V3 waveforms separately evaluate all failures,
actual optical power/pilotSNR/energy/time/constellations; no proxy fidelity claim.

Source-only USTAR/checksum, prerequisite hashes, additive conflict installer,
byte-exact handoff backup and atomic GitHub save. Local21 new nonTorch tests
and specified legacy regressions/source/restore checks in VERIFICATION.md;
mandatory10 Torch/HF toy tests and real saved BART/GPU recovery NOT executed
locally (unavailable), gated on server. No server task launched. GPU1 startup
16GiB/free4GiB reserve, disk8GiB, owned-child finite2h diagnostic cap. Rough
20-60min diagnostic/3-10min smoke UNMEASURED, refine measured progress; cap not ETA.

First user command: launch.py diagnostic in NEW experiment_results/v10_bart_latent_codec_v4_01,
width64. Return v10_bart_latent_codec_v4_records.tar.gz even native/overfit stop.
SCREEN_COMPLETE is execution, not quality success. Image PSNR/joint gains,
same-resource improvement vs old narrowV3, continuous-vs-digital/hardware and
publication remain unproven. See ../bart_latent_codec_v4/README.md.
