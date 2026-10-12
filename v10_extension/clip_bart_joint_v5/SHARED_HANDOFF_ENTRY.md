## 2026-10-13 — CLIP/BART C/I/T joint preparation V5

User requested history review, new joint module, training preparation and an
explanation before implementation. Old V1/V2/V3 text bridges failed clean gates;
V4 exact8/8 onTRAIN8 passed, not generalization. New clip_bart_joint_v5 only;
old sources/data/splits/weights/results preserved. No server task launched.

Same-caption CLIP semantics formC with frozen B4 image features; decodable V4
query codec formsT32x64; I224x16 frozen B4 route plus trainable TX/RX adapters.
C32x16/fixed32 BART memory. Frozen BART/CLIP/VAE/B4, zero-init receive fusion,
no original-word-position MSE, no source length/mask/UTF8 atRX, no diffusion.
Fresh P0 text codec1024train/3000batch4updates,32validation WER.60/dependence.10/
naturalEOS gate. Never loadTRAIN8 diagnosis weights as production initialization.

Three same-initial/registered-size/data/order/loss/LR/resource candidates:
J0 neural source/RX block-separated in SAME frame; J1 joint; J2 removeCLIP TEXT
SOURCE only, same CLIP TRAIN teacher remains. Own256pairs x4cleanepochs,
batch1/accum2, first200forward text-frozen then textLR2e-6; C/image/fusion1e-4,
private text anchor + C image/text semantics/coarse image task. All clean outputs
reported even negative. Source/RX separate control not optimal two-packet system.

Default screen clean only, independent full waveform evaluation atclean+6FSO.
Optional pipeline forks SAME each own clean checkpoint into matched clean and
FSO256x2 continuations only after own gate; Gamma-Gamma/pointing/pilot symbol
proxy omits waveform/temporality/DAC/MZM/sync, not link-matched fidelity claim.
Full fresh V3 waveform eval, actual lock/power/SNR/time/energy/constellations and
all errors retained. Czero/fusionoff/shuffle on same received frame plus wrong
source text diagnostics; not retrained noC baselines or free donor transmission.

C512+I3584+T2048=6144real/3072complex,R1. SAME V5 total frame/sampling contract,
AWG64GSa/s/4SPS/16GBaud/RRC.1, charged probe/pilot/preamble/tails/guards. More
payload than old2304, no old-budget benefit or equal actual energy claim. Images
useC/I, not source/decoded text prompts; no generated/diffusion image polishing.

Local20 new nonTorch checks +42 specified V4/COCO regressions, Python3.8/source/
release-restore checks recorded in VERIFICATION.md.10 new Torch/HF tiny contracts,
real saved backbones/GPU/training performance NOT locally executed, server gates.
GPU1 startup16GiB/free4GiB, disk12GiB, FP32, owned-child finite4h cap, no unrelated
process actions/environment changes. Rough30-90min clean screen/1-3h optional
noise pipeline UNMEASURED; use real logged speed, cap not ETA. No exact resume.

Source archive/checksum + additive conflict-checking installer, byte-exact
HANDOFF backup and atomic GitHub save required. First recommended user command:
launch.py screen NEW experiment_results/v10_clip_bart_joint_v5_01. It first
checks successful V4 screen identity, CPU tests and real small smoke, then fresh
prefit/gated joint screen. Return v10_clip_bart_joint_v5_records.tar.gz even
text_prefit_stop/clean_validation_stop. SCREEN_COMPLETE is execution, not
joint gain. No finaltest/seed sweep/digital/JPEG/hardware/publication claims.
See ../clip_bart_joint_v5/README.md for installation, input overrides and progress.
