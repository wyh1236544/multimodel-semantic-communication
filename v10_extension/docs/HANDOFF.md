# Handoff

## 2026-09-30 coherent waveform physical-link V2

New additive channel: see [../physical_link/HANDOFF.md](../physical_link/HANDOFF.md) and [../physical_link/README.md](../physical_link/README.md). No KK; actual bitstream/CRC/optional convolutional FEC/Gray QAM/RRC/64GSa/s. Old code and weights unchanged; real server/GPU validation pending.

## 2026-09-29 shuffled-Common evaluation repair

Server evidence from the user: the formal validation root has 27 planned complete evaluation directories plus `eval_validation_timing_mixed_seed42`; the freeze record does not exist. The old three `common_only_shuffled` runs are invalid because `eval_v10.py` evaluates one image at a time and the model rolled C32 along a batch dimension of one. A patch now caches each validation image's own pre-channel Common C32, feeds a deterministic different image's C32 into each target's single-image channel evaluation, records donor ID and payload distance, and rejects any batch-one shuffled call without an explicit donor. The patch also adds a dedicated new-output validation phase, strengthens the freeze gate, and includes no_uep/no_adapter/no_common_probe in test and aggregate. Existing checkpoints, split manifest, and all old results remain untouched. Local static/synthetic checks and real server CPU/GPU verification must be reported separately; do not claim test results from the patch.


## 2026-09-22 formal FP32 restart after AMP failure

The first server formal run on physical GPU 1 failed at mixed seed 42, epoch 1, step 140 with `FloatingPointError: Nonfinite gradient`. The formal launcher now passes `--no-amp` to smoke, training, validation and test evaluation, matching the previously successful bounded no-AMP pilot and keeping numerical policy consistent. Preserve `experiment_results/v10_formal_paper_v10_01` as failed-run evidence and restart under the new tag `paper_v10_02_noamp`; no result may be overwritten or silently resumed.

## 2026-09-22 preflight manifest reuse fix

`run_v10_formal_suite.sh preflight` now reuses the existing immutable `ln_v10_splits.json`, prints its SHA-256, and only creates it when absent. This fixes the safe-launcher `FileExistsError` without weakening the no-overwrite boundary. The current requested device is physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`).

## 2026-09-22 formal experiment closure update

The V10-A model architecture remains frozen. A formal experiment control layer was added: `run_v10_formal_suite.sh`, `docs/FORMAL_EXPERIMENTS.md`, image-cluster tail statistics in `eval_v10.py`, the existing shuffled-Common model path exposed in evaluation, and `aggregate_v10.py` for hierarchical paired bootstrap across at least three independently trained seeds. The formal launcher requires validation completion and an explicit freeze record before test evaluation. No real GPU training, COCO evaluation, server write, or performance claim was made in this update.

Static Python parsing, shell syntax, launcher dry-run, and a synthetic hierarchical-bootstrap check passed locally. The numerical evaluation path remains unexecuted locally because the available isolated runtimes do not contain PyTorch. Six of seven pre-existing deployment tests passed; the remaining test uses an external temporary directory that the current project write-boundary correctly rejects, and was not changed as part of this work.

Status: in-project deployment correction implemented; deployment-only tests executed locally; full numerical recheck and server smoke/performance pending. Last reviewed: 2026-09-20. This handoff supersedes the old sibling-directory deployment instructions, not the algorithm design.

Latest user boundary: all new code, outputs, logs and caches stay inside the existing `/data1/zw/WYH/multimodel`. Upload the new `.pyz` directly there and use its conflict-checking additive installer. Payload only adds `run_v10.sh`, `README_V10_INPLACE.md`, and `v10_extension/`. Existing root code, instructions, datasets, weights and results are not overwritten. Local staging and this turn's build/test outputs are under `work/multimodel`; prior source/release/snapshot preserved. No global changes or rollback of prior global changes is authorized by this revision.

The user authorized an isolated multi-channel/FSO research revision while preserving the current V9-A/V9-B.2 best artifacts. Baseline source: server_snapshot_20260920, extracted from the user-provided archive. No server connection, data images or weights are available locally. No server training is running under this task.

Implemented: complex AWGN/Rayleigh; nonnegative differential IM/DD FSO with Gamma-Gamma turbulence, pointing/path loss; pilot-based regularized equalization; noisy reverse control plus forward mode announcement; receiver-only deterministic token metadata; mode-specific fixed-power UEP; 2256-parameter residual receiver adapter; periodic true Common-only semantic training; AWGN replay; stage freezing and mixed/worst-group training; fixed image splits; per-image repeatable evaluation; paired image-bootstrap comparison; optional received-caption diffusion previews; bounded resource checks, atomic outputs, independent artifact verification, and a shell launcher.

Only the isolated copy of `models/joint_image_text_model.py` was changed among legacy core files: opt-in early-fusion bypass, image-gate repair, inactive-head bypass, and a separate RNG for random Common controls. Original 333 snapshot files were backed up, restored into an isolated verification directory, and compared by SHA-256. No real server code, weight or result was changed. No destructive cleanup performed.

Historical verification of the previous V10-A release used Windows Python3.12.14 + isolated torch2.6.0+cpu + PyYAML6.0.2, not the server's Python3.8/torch1.12. Its16 physics/synthetic tests passed. Those numerical algorithms and model files are byte-preserved here. This turn, the old dependency directories returned Windows access-denied and importing torch produced a namespace without `__version__`; read-only permission requests did not resolve it. No ACL edits or dependency reinstallation were performed. Therefore the full numerical suite was NOT successfully rerun in this revision. Failure logs remain in the local project's logs/v10. The new deployment-only suite uses stdlib, reports its suite name explicitly, and parses all extension Python with Python3.8 grammar. The server command `bash run_v10.sh check` still runs the full suite and must pass before real smoke. GPU AMP, real pretrained checkpoints and diffusion remain untested.

Local dependencies were installed only under the task workspace; an interrupted download/install was retried. No existing Python environment was upgraded. The test warnings about torch2.6 AMP deprecation are expected because production code deliberately retains the older torch1.12-compatible API. No local/server training remains running after verification.

Prior-turn history only: global Codex agreement files were saved under prior permission. This revision does not modify, delete or roll them back. The extension's AGENTS/CLAUDE entries now explicitly constrain edits to multimodel and point to the shared project agreement. Fresh-session/Claude Code loading was not independently tested.

Next: follow `../README_V10.md` to upload into `/data1/zw/WYH/multimodel`, verify hashes, run the FULL CPU suite, freeze split manifest including additional historical exclusions, and run GPU smoke (8 train images,4 validation images,1 epoch,30 minute cap). Then consider a2000-image pilot with a3-hour cap. No server job has been launched. Preserve V9 as baseline; do not initiate long formal training from this handoff alone. Do not reuse V9-B.2 AWGN verifier thresholds as calibrated FSO decisions. No improvement/newness/publication claims are supported until matched experiments are completed.

## 2026-10-02 physical-parameter optical-link V1

Current additive optical implementation: [../optical_link_v1/README.md](../optical_link_v1/README.md),
[verification](../optical_link_v1/VERIFICATION.md), [handoff](../optical_link_v1/HANDOFF.md).
User chose frozen token-guidance A as the main R&D baseline. New SI-unit coherent
FSO mode derives optical power/ASE/shot/thermal noise/SNR from configuration,
retains historical channels and C/I/T protocol, and adds no UTF8 side channel.
No neural training. Source-only tar.gz plus server install commands and atomic
GitHub synchronization are required for this release. Shared handoff update is
backed up; no historical code, weights or results replaced.

Local NumPy physics/protocol and regression checks plus random-byte diagnostics
are executed; actual existing A delta/base loading, Torch CPU hooks, GPU1/COCO
image/text evaluation and hardware calibration remain unexecuted here. No server
job or training remains running under this task. Next user-started steps:
optical_link_v1/launch.py check, link, 2-image smoke, then 8-image quick if passed.
Defaults are declared assumptions, not measured weather/device equivalence.

## 2026-10-03 final direction comparison V1

Additive module `v10_extension/final_compare_v1` implements frozen A validation
comparison of JPEG+exact UTF8 digital, semantic Q8/FEC/QAM, and the SAME normalized
semantic features sent continuously. Common 64GSa/s/4SPS SI coherent FSO, active
launch SIGNAL power and time/signal-energy ceilings; no free analog redundancy,
no original-text oracle in semantic receiver. Includes float/Q8/JPEG clean controls,
launch-power sweep (derived physical and pilot SNR), actual image/text/constellation
outputs, whole-matrix/packet/hash verification, image-cluster paired bootstrap and
strict result collection. GPU1; no training or automatic test-split selection.

Local NumPy optics/core/regression and source release/restore/additive conflict
checks are executed. Torch CPU frame contracts and actual GPU/COCO/base+A-delta
loading/effectiveness are NOT executed locally (Torch unavailable); user-started
pipeline gates smoke before 32-image/9-condition/3-channel-draw validation. No
server job started and no positive-benefit/publication claim. Preserve all old
modules, weights and results; source tar/checksum and atomic GitHub save required.

Next: follow final_compare_v1/README.md to install verified source, then launch.py
pipeline in a NEW project-local output directory. Return the completed collected
archive. Use validation to decide a direction, then separately lock parameters
before any future independent test. Frozen digital-A analog results do not prove
all trained analog/DeepJSCC approaches ineffective.

## 2026-10-05 optical realism V2

Additive `v10_extension/optical_realism_v2`: frozen A, no training, GPU1,
validation-only. SI optics plus clock-before-ADC/ENOB/IQ imperfections, public
interspersed-pilot timing/CFO/phase receiver for arbitrary complex payloads,
explicit rejection, fixed public-calibration laser working point, charged pilots
and repetitions, QPSK-convolutional vs 16QAM-convolutional-repeat2 comparisons.
Same device/peak-envelope/slot caps do NOT mean equal actual average power/energy.
Default pipeline is check -> 15 B2B diagnostics -> 2-image smoke -> TWO 8-image
quick runs -> verified image/text/constellation record bundle. Not formal test.

Local NumPy contracts/B2B/regression checks executed; actual Torch hooks,
base+A delta loading, GPU/COCO performance, hardware calibration NOT executed
locally. No server tasks launched. Historical source/weights/results preserved.
Source tar + checksum + additive installer + GitHub atomic backup required.
See optical_realism_v2/README.md and VERIFICATION.md for commands and limits.
First return: v10_optical_realism_v2_records.tar.gz from a NEW pipeline output.

## 2026-10-07 optical realism V3 receiver repair

Additive optical_realism_v3 preserves V2, historical weights/results and split.
Public even-pilot coherent bounded CFO search (default +/-250MHz), joint
buffered clock fit, smoothed complex pilot phase and odd-pilot validation replace
noisy unwrap/random-walk acquisition. The synchronization score has a new
held-out coherent definition, not merely a lower V2 threshold. No hidden
CFO/clock/fading/source-feature access.

Analog reliability retains an unscaled held-out block-error floor and split-half
repetition inconsistency as well as white variance/R. It is an uncalibrated
conservative heuristic, not certified delivery. Whole-coded-block 16QAM repeat/
LLR combining replaces adjacent pairs: all 16 points are reachable, all repeated
bits/pilots/samples charged. JSON/Markdown and effective preset identity repaired;
pre/post-gate constellations, separate waveform vs RX-selected projection SNR,
probe/payload lock and accepted-but-MSE>1 diagnostics retained.

Local 140 NumPy/protocol/legacy regression tests, 15 B2B functional cases and
24 same-ADC synthetic V2/V3 comparisons executed (exit0). Combined moderate
synthetic mean feature MSE 1.269 -> .066; stress .753 -> .276; good .00354 -> .00367
(not uniformly improved). No V3 accepted MSE>1 in this small synthetic set.
No Torch, real A/COCO/GPU1, V3 image/text gain or hardware calibration executed
locally. Original A text ceiling unchanged.

Source-only tar/checksum, restore/install/conflict checks, additive installer and
atomic GitHub save required; installation backs up HANDOFF before append.
Next user-started step: optical_realism_v3/launch.py pipeline in a NEW project
output, GPU1, --max-hours 1: tests/Torch hook -> B2B -> synthetic replay ->
2-image smoke -> two 8-image quick contrasts -> verified record bundle.
No training, no seed sweep, no automatic full validation/test. Return
v10_optical_realism_v3_records.tar.gz, then decide on middle power points.
No server jobs launched by this task. See optical_realism_v3/README.md and
VERIFICATION.md for commands and limits.

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

## 2026-10-11 frozen-weight expanded FSO JSCC evaluation V1

Additive fso_jscc_eval_v1 evaluates the EXACT four completed fso_jscc_v1 screen
weights. NO optimizer/retraining, no environment upgrade, physical GPU1 only.
128 fixed validation-prefix images: pilot32 vs additional96 reported separately.
Additional means unused in that 32-image screen, not guaranteed unseen in all
historical experiments. No final test/calibration split is used. One training
replicate; channel draws are independent trials, not combined transmissions.

R=1, unchanged token/symbol budget, AWG64GSa/s, 4SPS, RRC.1 and V3 receiver.
Proxy25,088 plus waveform6,144 =31,232 rows. Six physical points, two independent
draws. Predeclared primary: M3-M2 PSNR on new96, per-image average over
moderate_p24/p21/p18 and draws, image-cluster bootstrap. Failures retained;
pilot replay, strict delta identity, before/after file and full model state hashes
checked. Optical power, offline projection SNR, pilot SNR, energy, text metrics,
images/constellations included. No analog BER or oracle text side channel.

Public slot/samples/peak-envelope limits match, not exact actual launch energy.
This is validation confirmation of FSO-vs-AWGN fine tuning, NOT proof of joint-vs-
separate, analog-vs-digital superiority, independent semantic fidelity or hardware.
Local NumPy/source/archive tests passed; Torch/real checkpoints/GPU unexecuted
locally (Torch unavailable). User-started launcher gates CPU tests and two-image
frozen-weight smoke before expanded evaluation; displays measured time projection.
Default 6h safety cap is NOT ETA. No server task launched by this code delivery.
Return v10_fso_jscc_eval_v1_records.tar.gz. See fso_jscc_eval_v1/README.md.

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

## 2026-10-12 text-only source/codec/AR recovery screening V1

Additive text_jscc_v1, physical GPU1,FP32, existing environment. No old source,
weights/results replaced; joint image-text deployment remains unchanged. Source
reader accesses only fixed-split caption JSONL, not image pixels. Frozen exact
B4/base/A/B2 reference is loaded (VAE instantiated but never run/train here).

Two same-initialized old AR-head TRAIN diagnostics: precompression versus clean
B4 Ctext16+T32 memory; matched/zero/shuffled actual greedy output. New native
token-embedding/Transformer/query-compressor/TX/RX/AR branch overfits32 texts,
800updates, requires<=.10 coveredWER and>=.10 zero/shuffle deterioration. Main
starts from original initialization, NOT diagnostic fitted weights.2000texts,
8epochs,batch4=4000updates;64 validation<=.20 coveredWER plus memory-dependence
gate before FSO. Overfit/clean gate failures yield explicit diagnostic-stop
archives, not main COMPLETE, no threshold relaxation. Smoke bypass quality is
explicit execution-only. Existing CLIP tokenizer case/spacing normalization and
128-token truncation recorded; not byte-lossless UTF8 transmission.

True text-only48x16=768reals/384complex,R=1; no image payload/raw-text bypass.
New short-frame public-waveform train16/holdout8 calibration per condition,
NOT slicing the old2304bank. Current learned payload fresh-waveform holdout
checks at clean/T1/T2. Stochastic five-tap widely-linear replay is approximate,
not full differentiable hardware. CompleteV3 waveforms score final results;
64GSa/s/4SPS/16GBaud/RRC.1, old optics/weather unchanged. Probe/tracking/tails/
guards all charged. Actual power/SNR/energy/time and constellations preserved.

Paired T1 clean-continuation/T2 FSO start from sameT0, same2000texts×2epochs,
1000updates,size/LR/order;25%clean anchors inT2. Real gradients AND parameter
updates required in source/compressor/TX/RX/decoder; checkpoint reload and old
model hashes checked. Final L0frozenB4text/T0/T1/T2×64×7=1792rows, failures kept.
L0/new is whole-architecture reference, not matched-parameter single-change
causal evidence. Image-cluster CIs conditional on one training/fixed weather,
historicalvalidation; no finaltest/digital superiority/paper/hardware claim.

Local31 non-Torch contracts and NumPy short-waveform tests, Python3.8 source,
release/deployment/restore checks recorded in VERIFICATION.md. Torch CPU8toy
contracts/pretrained/GPU/actual text improvements UNEXECUTED locally. No server
job started. Tentative5–15min smoke,1–3hmain planning range;45min/4h are STOP
budgets, not ETA. User launcher checks real Torch and smoke before formal main.
Return v10_text_jscc_v1_records.tar.gz even on diagnostic_stop; large weights
stay on server. Additive source tar/checksum, recoverable HANDOFF append and
GitHub save required. See ../text_jscc_v1/README.md.

## 2026-10-12 text-only JSCC frozen-decoder initialization repair Fix1

User's server evidence: original8 Torch toy tests passed, but real GPU smoke
failed native_overfit learning audit with46.09GiB free. Frozen old model
initialization exposed an implementation defect: ARTextDecoder deep-copied
old.decoder and retained requires_grad=False. Original toy fixture did not
freeze its reference, so could not cover that production path.

Additive text_jscc_v1_fix1 keeps text_jscc_v1 and failed outputs unchanged.
Only the independent candidate language copy is explicitly trainable; original
CLIP/B4/VAE remain frozen/unaliased and their hashes/gradient checks remain.
All text-frame/model/loss/data/schedule/quality/fidelity thresholds stay the
same. No removed audit, longer-training workaround or expanded unfreeze.

Training now writes per-prefix trainability before optimization and refuses
frozen required candidate groups early. learning_audit.json records exact
gradient/update failures before raising. Torch toy fixture now uses frozen
references in all8 existing tests; adds2 explicit frozen-copy trainability/
no-alias/old-weight preservation and real decoder-gradient/update regressions.
Server must rerun all10 Torch tests and real short smoke. Local31 non-Torch
contracts/source/deployment checks are separate; Torch/real repair effectiveness
UNEXECUTED locally. No server training launched here.

81>77 tokenizer warning is not this RuntimeError: raw tokenization may exceed
CLIP's77 capacity but CLIP forward already uses<=75content+BOS/EOS chunks.
Do not silently truncate original texts or enlarge CLIP positional embedding
to hide the warning.128-token candidate coverage contract remains unchanged.

Same GPU1/FP32/384complex/R=1, fresh output
experiment_results/v10_text_jscc_v1_fix1_01; old _01 retained. New additive source
tar/checksum and GitHub save, HANDOFF byte-backup then append required.
Return v10_text_jscc_v1_fix1_records.tar.gz after completion/quality stop; on
execution failure return smoke log, FAILED and available training audit.
See ../text_jscc_v1_fix1/README.md. No recovery-improvement/hardware claim.

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
