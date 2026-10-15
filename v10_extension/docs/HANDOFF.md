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

## 2026-10-12 — additive original COCO caption and single-modality BART V2

User requested original short-text dataset and independent image/text transmission
on current CLIP+BART+VAE architecture, source archive, commands and GitHub save.
New bart_coco_modal_v2 only; old source/results/weights/LN labels/split preserved.
Official captions_train2017.json/captions_val2017.json must exist on offline server;
read-only data-check checks original image IDs, selects minimum annotation ID,
independent hashed caption views. No final test/calibration business access.

Default independently trained image_only C32/I224=2048complex and text_only
C32/T32=512complex; optional joint full2304. Opposite source never enters encoder;
text-only never opens image pixels; image-only legacy B4 receives constant dummy
text, not actual caption. Same registered V1 bridge, different used parameters;
frozen CLIP/BART/VAE/exact B4. RX public mode/pilot/compact features only, no GT,
source mask/length/UTF8 bypass. Decoded text-to-image path remains deferred.

Clean training ONLY then real compact full V3 optical waveforms, not new noise
training or sliced2304 proxy bank. 64GSa/s/4SPS/16GBaud/RRC.1/R1, probe/preamble/
pilot/tails/guards all charged; 2048/512/2304 payload uses12026/5114/13178samples.
Same peak/slot caps, not equal actual energy or equal-resource causal comparison.
No invented analog BER/CRC/bitrate; untransmitted task metrics null.

Text short T32 TRAIN diagnostic8/1200forward/200warmup, then own C+T diagnostic
same budget; private failure does not forbid C+T. Actual mode gate WER<=.20 and
zero/shuffle margins>=.10; reset initial bridge for256x4 clean main if passes.
Validation16 quality gate WER<=.60 +same margins, no checkpoint selection.
Image clean main independent of negative text quality gate; engineering failures
stop and preserve FAILED. Same-grad/no-grad actual forward tolerance.0002 audited
before each stage, no silently changing kernels or suppressing errors.

Each evaluated mode112rows(clean+6FSO)x16; fixed first4 image/text/constellation
examples, all erased frames counted. Actual C-zero/C-shuffle receive sensitivity
controls not independently trained best-private baselines. COMPLETE only all
requested modes evaluated, PARTIAL_COMPLETE negative text quality stop, SMOKE
only execution. Report recomputes text metrics and verifies matrix/resources,
training groups/frozen updates, hashes; weights stay server, records archive small.

Source/install/restore checks +21 new nonTorch tests +49 inherited NumPy tests
passed locally; new11 Torch/HF contracts and real pretrained GPU gates SERVER
only, not locally executed. PhysicalGPU1 free>=16GiB/reserve4GiB, existing env
unchanged. Plan20-60min UNMEASURED, finite2h cap not ETA. No server job launched.
Default launch.py pipeline experiment_results/v10_bart_coco_modal_v2_01
--max-hours2 (space arguments per README). Send v10_bart_coco_modal_v2_records.tar.gz
even PARTIAL; missing originalCOCO JSON/engineering failure send exact log first.
HANDOFF exact-byte backup before append. See ../bart_coco_modal_v2/README.md.

## 2026-10-12 — additive BART text recovery interface/LoRA screen V3

User authorized implementation following failed clean COCO short-text V2.
New text_recovery_v3 only; all historical sources, weights, datasets, fixed image
split/results preserved. Actual BART encoder/cache/teacher-vs-incremental logits
checked first. Masked/unmasked/pad-zero native source-memory oracles are TRAIN
diagnostics, not transmitted methods. Conditional native LoRA overfit is reset
before production candidates. No image/VAE/B4 checkpoint loaded or trained;
multimodal C/I/T integration and actual recovered-text image guidance remain
future work, not claimed executed by this text-only repair.

New explicit length protocol: seven bits (BOS/EOS length-1), two symbols each,
two public padding reals =16 real/8 complex charged same-channel header; no CRC,
no free original source mask/length at RX. Wrong accepted lengths retained in
physical diagnostics. No raw UTF8 payload; R=1 whole frame (header redundancy
does not mean whole-feature repeats). A0 legacy privateT32=256complex; A1 mask
=264; A2 optimization bundle same264; A3 same plus rank8 last-two-layer BART
decoder cross-attention Q/V LoRA49,152 parameters; A4 width64=1032complex capacity
diagnostic, not same-resource gain. Optional A5 text-onlyC32/T32=520complex;
not image-text joint evidence or equal-resource C gain. Old V2 remains runnable.

Default diagnostic: five matched-seed TRAIN8 arms,1200forward/600optimizer each,
same initial bridge for narrow arms. A0/A1 retain old loss/LR; A2+ ramp CE after
200warmup, stronger valid/pad feature supervision, lower bridge LR. Fixed last
step, snapshots actual greedy/EOS, zero/donor memory interventions, no repetition
ban/forcedEOS/threshold relaxation. WER .20 + memory margins .10 retained, plus
natural EOS gate. A3-A2 isolates light decoder adaptation at same payload/data/
update budget; extra compute/trainable parameters logged, not claimed equal.

Optional pipeline expands only own-passing A2/A3 from UNTRAINED initialization:
256train x8 cleanepochs,32 fixed validation, WER .60+memory .10+natural EOS gate,
then fresh complete V3 compact waveforms for six FSO conditions. No new noisy
training, no old2304 proxy slicing, no test/calibration business data. Header,
probe/pilot/preamble/tails/guards charged; actual optical power/energy/SNR,
failure/length errors/text/constellations retained. Same slot/peak cap not exact
energy. No analog binary BER, digital/JPEG superiority or publication guarantee.

GPU1,FP32,batch1/accum2,existing Py3.8/Torch1.12/TF4.30.2 environment; no install
or model download.16GiB startup/4GiB reserve; owned-child finite stop budget,
no unrelated process actions. Local nonTorch/protocol/physics/deployment tests
listed in VERIFICATION.md; Torch tiny numerical/actual pretrained GPU recovery
NOT_EXECUTED locally. No server job launched. Rough15-45min diagnostic is a
planning range based on old V2 timings, not measured V3;2h cap not ETA.

Source tar/checksum/additive installer, byte-exact shared HANDOFF backup and
GitHub atomic save required. First user command: launch.py diagnostic in NEW
experiment_results/v10_text_recovery_v3_01, then return collected
v10_text_recovery_v3_records.tar.gz. SCREEN_COMPLETE means screen executed,
not successful recovery; negative/native-stop outcomes remain visible.
See ../text_recovery_v3/README.md. Weights remain server; no exact resume.

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

## 2026-10-13 — CLIP/BART joint V5 fix1: secondary beam records

User-approved report-only repair after server smoke completed all three arms
but finish rejected V4 matched_beam4 rows. No CUDA OOM observed (46.90GiB free).
V4 private_controls writes matched/zero/shuffled/matched_beam4; joint controls
write only three. Prefit verifier now explicitly requires all four, recomputes
all text metrics, rejects duplicate/missing/unknown/split errors, and returns
only original three greedy controls for unchanged quality gates. Beam decode
cannot rescue a failed greedy gate. No training/network/channel/config changes.

Eight new regression tests cover actual four-control schema and negative cases;
20 report tests pass locally. No new server GPU/training verification performed.
Patch install preflights known old/new source hashes and all historical deps,
backs up report/test/manifest/HANDOFF before replacement, and is idempotent.
Original release archive and failed experiment_results/v10_clip_bart_joint_v5_01
stay unchanged. Rerun screen in NEW v10_clip_bart_joint_v5_02 (full smoke first).
Do not delete FAILED.json or reseal historical failed records as success.
Deployment and actual verification details: ../clip_bart_joint_v5_fix1/README.md.

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

## 2026-10-13 — CLIP/BART joint V6 fix1: clean SNR verification

User's real GPU smoke executed all three arms and both evaluated conditions, then
finish rejected receiver-condition NPZ SNR. GPU free28.27GiB; no OOM in supplied
log. Unchanged V5 clean_args supplies40dB as FINITE CLEAN CONDITIONING SENTINEL,
not a measured physical/pilot SNR. V6 evaluator faithfully saves it, but report
expected20dB. Original synthetic fixture duplicated wrong20 and hid mismatch.

User-authorized report-only repair: expected clean40, retain finite checks and
1e-4 physical float32 tolerance; FSO still compares actual recorded pilot SNR.
Error now identifies condition/image/expected/stored. No receiver/model/source/
training/schedule/loss/resource/FSO parameters or quality thresholds changed.
32 existing reports +7 new cases: production clean_args body NumPy shim and
actual Torch interface, clean20/zero rejection, physical float32 match/mismatch,
nonfinite rejection. Local38 pass,1 real-Torch case skipped (Torch unavailable);
server must run that case plus existing26 Torch/HF tests and new real smoke.
New regression reproduced original failure BEFORE fix; full positive clean/
paired-FSO matrices passed AFTER fix. This is not proof of improved recovery.

Hash-locked fix1 installer preflights all replacements/unchanged module and378
historical sources, backs up report/test/manifest/HANDOFF before replacement,
rejects unknown/mixed/symlink/drift, idempotent. Old V6 archive and failed
experiment_results/v10_clip_bart_joint_v6_01 remain unchanged; do not delete
FAILED or reseal old results. New run v10_clip_bart_joint_v6_02, GPU1, mandatory
check -> real smoke -> formal screen. No server job launched locally.
Source tar/checksum and GitHub save required; deployment/verification commands
in ../clip_bart_joint_v6_fix1/README.md.

## 2026-10-13 — CLIP/BART joint V6 fix2: independent TRAIN caption identity

User-approved follow-up after fix1 real GPU smoke passed SNR checks but finish
rejected Adapter TRAIN teacher differs from declared source. V2 joint cache's
source_sha256 hashes image float32 bytes PLUS original caption; previous verifier
incorrectly compared caption-only SHA to that joint SHA. Old synthetic fixture
used text-only digest as fake joint identity and hid the production mismatch.

Keep all original joint source hashes and normalized original captions unchanged.
V6 attach_text_source_hashes adds separate text_source_sha256 to image receipts,
cross-checking earlier independent BART caption-view hashes, fixed image IDs,
selected annotation IDs/file SHA/split/UTF8 length. Runs immediately after real
TRAIN/validation cache and before joint training. Audit metadata only, never
receiver/source tensor/length side channel. Generation teacher verifier uses
text-only digest AND independent BART TRAIN receipt, fixed IDs/split, full vector
counts; no silent short-vector zip or acceptance of joint hash as text hash.

Only V6 core/run/report/test_core/test_report/MANIFEST replacement. Model,
receiver_cache NPZ producer/schema, training/channel/config/budget/quality gates
and fix1 SNR correction unchanged. Tests execute original production joint-hash
expression using a NumPy interface shim, not a claimed Torch/GPU cache. New
realistic fixture reproduced original verifier error before repair. Local18core
and43report pass;1 actual Torch case skipped, still required server;49 historical
V5/V4 tests and8 deployment contracts run separately in release verification.
This validates metadata, NOT recovery gains or real saved-backbone execution.

Fix2 requires verified V6+fix1 installed; exact six source backups and HANDOFF,
all378 old dependencies checked, unknown/mixed edits refused, idempotent. Keep
immutable V6/fix1 archives and failed_01/_02 records unchanged. New_03 executes
checks -> real smoke -> own-gated formal screen, GPU1; no server job launched
locally, no failed-run reseal or checkpoint reuse. Source tar/checksum and atomic
GitHub save required. See ../clip_bart_joint_v6_fix2/README.md.

## V10 bidirectional RX V7 — additive receiver-only test (2026-10-13)

New `bidirectional_rx_v7`, physical GPU1, existing offline env only. Preserve all V6/fix2/older code, data, weights and results. Source baseline = verified formal V6 J0 C/I/RX + V6 private text global9000; live B0 measured, not historical replay. Freeze all old modules. Four equally registered initialized receiver adapters: self-only, T→I, I→T, bidirectional. Matched clean/FSO symbol-proxy training; full-waveform evaluation generated once per condition/pair and reused across all receivers and controls. Raw source text/images/masks/length are loss/evaluation only. C/I/T3072 complex symbols/R1 unchanged. Synthetic private feature drop is separately labelled, full resource charged. Primary compare R3−R0 in moderate physical FSO with outages retained; one seed/small reused validation, not independent-test proof. Start diagnostic32/VAL8, not full pipeline; no auto promotion or dependency changes. Local Torch/GPU unavailable; CPU record tests do not confirm saved pretrained smoke. No diffusion, no Common/source retraining or compression-savings claim. See module README for exact packaging/run/ETA/caveats.

## DOCCI / Long-CLIP / larger-image C/I/T V8 (2026-10-13)

User explicitly replaces current dataset with DOCCI and requests Long-CLIP plus larger image input. New independent `v10_extension/docci_longclip_joint_v8`; preserve all historical COCO/splits/weights/results/modules. Module launch is independent of legacy root run_v10.sh.

Official DOCCI train/qual_dev only, original full descriptions, no truncation/filter or test/qual_test training/model inputs. Native full view letterboxed512, valid rectangle loss/eval only;256 smoke before mandatory512 authentic backward smoke. BART max512 tokens default (audit selectable256/1024), T64×64 learned memory; native248 Long-CLIP-B full ordered chunks with author implementation/licenses and official599MB SHA256 pin. NativeLongCLIP image224 remains separate from512 VAE/spatialJSCC path. BART/VAE/LongCLIP frozen; new spatial encoder/decoder, Common and TX/RX fusion learned.

Paired separate vs joint registered parameters and C512/I8192/T4096 real budget at512,6400 complex symbolsR1; no rawUTF8 text, no source length/mask or clean semantic features atRX. Separate common halves and private receivers are modality-isolated; joint cross-modalCommon/RX. Additional same-received-frame zeroCommon/fusionOff/receivedwrongC/I/T controls. Proxy FSO training and full assumedV3 optical DSP evaluation are labelled differently; no diffusion or newdigitalbaseline, no pure causalcleanvsFSO claim with unmatched continuation updates. Actual waveform/energy/compute are not equal merely because public caps/register counts are equal.

GPU1, batch1/gradaccum2FP32, existing offlineTorch1.12/Transformers4.30.2/Diffusers0.20.2; no dependency changes or otherprocess stops. Default diagnosticTRAIN32/VAL8 text400/image128/armclean128+FSO64, then screenTRAIN128/VAL16 if private long-text gate passes. Gate failure is QUALITY_STOP and still collectable; do not automatically lengthen joint training/lowergate. SMOKE8-step checkpoints never initialize main. Overall caps are not timing benchmarks; log stageETA and allocated/reserved memory after actual steps.

Source-only tar/checksum, all-conflict additive installer and exact HANDOFF backup before append. Local syntheticNumPy/data/report/full6400 optical frame/archive checks only; local Torch/pretrained/GPU/DOCCI quality and speed NOT_EXECUTED and required servergates. Official DOCCI/nativeimages and LongCLIPweight must be uploaded separately, see DOWNLOADS.md. Collect only lightverifiedrecords; keepweights/nativeimages/cache server. Every published source revision is saved atomically to privateGitHub main with expected-head lease, never force push.

## 2026-10-13 DOCCI independent private branches V9

User requests first making separate text and image branches work after DOCCI/Long-CLIP/512 migration. Added `docci_modal_v9` only; historical data/source/checkpoints/results preserved. This temporary baseline does not abandon later Common/C-I-T experiments. Text uses frozen BART and V6 completed P0 private codec,32x64 slots, old128 source offsets preserved and new offsets zero/trainable to512. Image uses frozen VAE, native512 full-view and only V8 completed image encoder/decoder. No Common/CLIP/Long-CLIP/diffusion/raw UTF8 atRX; modality backbones load one at a time. Default diagnostic: actual backbone/GPU smoke, native source oracle separated, zero-shot historical DOCCI validation, TRAIN4 overfit text1200 optimizer updates batch4/image256updates accumulation2, independent fixed qual_dev8 full recovery and zero/shuffle controls. Bad text quality does not prevent image diagnostic; engineering exceptions remain fatal with logs. Production screen resets initial weights, not fitted diagnostic weights. Explicit pipeline has matched clean/noisy continuation and actual V3 R1 private short-frame evaluation. Text1024/image4096 complex payload symbols; not equal-budget joint gain.

Source-only archive `v10_docci_modal_v9.tar.gz` unpacks to `_v10_docci_modal_v9`; invoke its `install.py --project-root .` before module launch. Existing server Python3.8/Torch1.12/Transformers4.30.2/Diffusers0.20.2 and physicalGPU1, no dependency changes/download or unrelated process control. Data root defaults project root, notDOCCI subfolder. Launcher diagnostic new `experiment_results/v10_docci_modal_v9_diag_01 --branch both`; automatically collects `v10_docci_modal_v9_records.tar.gz` under output. Local stdlib/NumPy/restore contracts only; authentic Torch/pretrained/GPU results NOT_EXECUTED locally. Engineering completion ≠ recovery success. Only TRAIN/qual_dev inputs; no test for model choice. Shared HANDOFF exact original-byte backup before append; no older entry rewrites.

## 2026-10-14 DOCCI reconstruction-first V10

User authorizes implementation after V9 full-view512 latent-primary recovery~19.42dB and long-text WER~198.6% on fixedqual_dev8; native VAE~26.89dB / native BART~3.99% are untransmitted references, not resource-matched bounds. New docci_reconstruction_v10 only; preserve all old modules/data/checkpoints/results. Frozen existing BART/VAE, physicalGPU1, offline FP32, no environment/new-model download/unrelated process control. Independent branches first, no Common/Long-CLIP/diffusion claim.

I0 old latent-primary vs I1 pixel-primary share exact old V8 image initialization, architecture, data/order/LR/updates and4096complex payload. I2 own64/96/128 multiscale alternating shifted-window/relative-offset/cyclic-wrap-mask VAE-latent codec uses same payload but fresh weights/more parameters, NOT isolated architecture causal effect or SwinJSCC reproduction. Record weighted-loss gradients at received latent, not value-fraction or total-parameter gradient fraction. Explicit image widths8/16/32 yield4096/8192/16384complex; rate changes require fresh init and charge time/energy.

T0 whole512 BART description ->32x64; T1 complete original ordered BPE chunks62content+BOS/EOS64, shared codec nine PUBLIC fixed slots. ALL empty slots encoded/sent/decoded/charged; no free source count/length/mask or original IDs atRX. Nine-segment R1 is NOT nine repeats;9216 vs1024complex must not be advertised equal-rate compression gain. Concatenate generated content BPE IDs, then decode once without inserted spaces; unrestricted greedy/EOS, empty-slot hallucinations/capacity stops retained. Whole native memory and native segmented source-mask ORACLE references explicitly separated from transmission. Frozen TRAIN targets only for optional noise consistency.

Default diagnostic genuinely trains64 different TRAIN inputs perarm (text1200updatesbatch1; image512updatesmicro1accum2), fixedqual_dev16 and TRAINfirst8 gates; no TRAIN4-only fit, validation-best selection, smoke reuse or auto promotion. CPU tests and ALL requested real256/512 smoke gates before substantive training; quality failures archive and continue siblings, engineering failures fatal preserve logs. Screen/pipeline are explicit new runs. Own-passing clean arms can fork equal-update clean/FSO proxy continuation, then full oldV3 optical waveform eval with actual full new frames, failures/power/SNR/time/energy retained. Proxy is not hardware/full-DSP differentiable training. AWG64G/4SPS/16GBaud/RRC.1/R1; no analog binaryBER invented. Same caps do not equal spent energy.

Source tar v10_docci_reconstruction_v10.tar.gz stages _v10_docci_reconstruction_v10; conflict-checking install.py before module launch. Exact old HANDOFF backup then append, source-only checksum and atomic GitHub save required. Local13 nonTorch tests and full9216-symbol NumPy waveform/source/restore tests; mandatory10 Torch/HF tests plus realBART/VAE/GPU/DOCCI effects NOT_EXECUTED locally. No server job launched. First user-started launch.py diagnostic NEW experiment_results/v10_docci_reconstruction_v10_diag_01 --branch both --max-hours4 (arguments spaced per README); return collected v10_docci_reconstruction_v10_records.tar.gz.30–120min UNMEASURED planning range,4h stop budget notETA. See moduleREADME/SOURCES/VERIFICATION; limitations and all negative outcomes preserved.

## 2026-10-14 DOCCI explicit pure-CE weight continuation V11

User approves continuing after approved TRAIN2048/batch2/10000 V10 record review:
same16 validation WER84.71%->49.94%, all10049.02%, not full recovery. New additive
docci_text_continue_v11 only; old modules, weights, data, quality gates/results
unchanged. Source-only tar/checksum/additive installer and atomic GitHub save.

Exact approved parent record/checkpoint hashes pinned. Genuine offline BART/GPU1
backward/save/reload smoke, then restore parent EXACTLY; no smoke reuse. NEW AdamW
and phase LR3e-5->1e-5, not exact optimizer resume or a retrospective20epoch LR
schedule. Pure CE additional10480 steps to cumulative20480/20epochs; source order
uses original global cursor10000, same2048 officialTRAIN/100qual_dev, batch2,
32x64 shared codec/all9 public chunks/9216complex/R1. BART frozen; no new model,
environment upgrade, CLIP/Common/image/diffusion/FSO or scheduled sampling.

Fixed validation/TRAIN8 probes every2000 additional steps; finalTRAIN32/VAL100;
fixed last, no validation-best selection or quality-based early termination.
Gold/native memory and self-prefix diagnostics are POST-HOC ONLY, never actual
receiver inputs/loss/gates. Measure content CE/accuracy, EOS calibration and
common-prefix positions separately. EOS after62 content BPE is not truncation;
capacity stop is not repetition. All source lengths are evaluation-only.

Exact Myers bit-vector Levenshtein keeps historical WER/CER/normalization/BLEU
definitions; real prior-row regression plus randomized Unicode/token tests.
One independent full CPU matrix/frame/metric/history verifier in collect, no
duplicate standalone report or180s cap; bounded overall owned-child budget and
progress retained. Old QUALITY_STOP/CLEAN_READY computed unchanged; COMPLETE
means engineering completion, not model quality. New checkpoints save optimizer
and RNG too, but this release does not expose exact-resume of a failed new phase.

Local NumPy/stdlib/arithmetic/negative reporting/source/restore checks described
in VERIFICATION.md. Actual Torch/HF contracts/approved checkpoint/BART/GPU smoke
and recovery gain are NOT locally executed (runtime lacks Torch/Transformers).
Mandatory server CPU+real smoke before main; no remote job started by delivery.
GPU1 startup16GiB/free4GiB reserve, project disk12GiB; no unrelated process control.
Extra pure training roughly16min from observed old speed, not total ETA; probes,
validation/checkpoint IO additional.4h overall safety cap, not timing benchmark.

Upload v10_docci_text_continue_v11_fix1.tar.gz and sha256 into existing multimodel;
extract _v10_docci_text_continue_v11, run install.py --project-root ., then
launch.py pipeline NEW experiment_results/v10_docci_text_continue_v11_01.
Default parent is v10_text_ce_2048_20261014_171836/screen_T1_chunks; missing or
different approved record/weight aborts, never silently use V6/diagnostic weights.
Return v10_docci_text_continue_v11_records.tar.gz; large checkpoints stay server.

## 2026-10-14 V11 CPU-only immutable collection recovery

User's completed run v10_docci_text_continue_v11_20261014_201245 reached all100 validation
and RUN_FINISHED, then collector rejected prefix FP32 weighted aggregates. Read-only
user audit found31 strict1e-7 failures; five printed errors5.83e-8..3.32e-7 support
rounding mismatch. Only those five scalar examples are locally available, not all31
raw rows. Never claim all raw records passed before the server reruns full verification.

New additive docci_text_collect_fix_v11 only. Old V11 code/MANIFEST/source_snapshot,
diagnostic values, data, checkpoints, failed output and logs remain byte-preserved.
No Torch import, pickle loading, GPU, training or generation. Only diagnostic function
is scoped to a declared new verifier; every nonaggregation validator AST matches old,
other comparisons/quality WER/EOS/margin gates unchanged and restored even on failure.

Finite nonnegative original FP32 means, exact public integer counts<=510/slot<=62;
error allowance computed from FP32 reduction roundoff bounds, not observed fitted
threshold. Full raw/hash/frame/matrix/history/saved-checkpoint checks still execute.
Actual differences, counts, allowance and prior-strict rejection flags retained in audit.
Excess discrepancy/hash/wire/count/scope/nonfinite failures remain fatal.

Full verify original first, copy only light records with before/after SHA checks into
NEW separate experiment_results/v10_docci_text_continue_v11_20261014_201245_recollect02,
then same legacy summary/HTML and new light archive. Original run gets no rewritten
COMPLETE/summary or other writes. Source fix snapshot and explicit same-neural-run
receipt archived. Partial output preserved; use recollect03, never overwrite retry.

Existing offline Python3.8/NumPy only; no dependency installation/network/new model.
16 local numeric/structural/copy tests plus old19 CPU regressions and isolated install/
source/USTAR/restore conflicts; real raw31/all-run server verification NOT_EXECUTED
locally. CPU collection cap30min is not ETA. Source tar/checksum and atomic GitHub
save required. See module README; no server task launched by delivery.

## 2026-10-14 V11 uploaded real results verified; clean text still QUALITY_STOP

This entry supersedes the earlier collection-recovery entry's "raw31 not locally
executed" status only. Historical entries remain byte-preserved. User now uploaded
v10_docci_text_continue_v11_records_collect02.tar.gz, SHA256
1dc0e70bb758aa304940145d3f3b7f06e41d3e8954381db54d419eee1382f01f.
Run: v10_docci_text_continue_v11_20261014_201245; recovered light output:
v10_docci_text_continue_v11_20261014_201245_recollect02. Complete trusted CPU
matrix/frame/metrics/history/source validation passed;615 original light files
match byte/hash identity;280 weighted diagnostic checks passed, including31
legacy strict rejections; max difference5.103898068625767e-7, max bound ratio
0.00945483928156349. No uploaded scripts executed as trusted verifier, no local
GPU inference/checkpoint-binary loading, no new server job. Original run remains
immutable. COMPLETE denotes engineering verification, not successful recovery.

Canonical result and next-step note:
[results/v11_text_continue_20261014/README.md](results/v11_text_continue_20261014/README.md).
Quick cold-start entry: [CURRENT_STATUS.md](CURRENT_STATUS.md).
Aggregate JSON, CPU verification receipt and corrected vector summary are included;
no raw DOCCI descriptions/images, full example HTML, caches, records archive or
weights added to Git. This is a documentation-only update, no training/source
contract/quality gate change. Exact old HANDOFF backup:
../deployment_records/HANDOFF_before_v11_results_20261014.md.

Same TRAIN2048/batch2/qual_dev100, BART frozen/private codec, pure CE clean,
all9 public fixed chunks /9216complex/R1. NEW AdamW/LR phase, not exact resume.
Fixed final weights at20480, no validation-best selection. Macro validation WER
49.0179566%->46.2205988%, SD6.38pp->7.14pp across descriptions; delta-2.7973578pp,
relative5.71%; paired conditional description-bootstrap95%CI[-3.6050415,-2.0044208]pp.
72 improve/6 tie/22 worsen; whole exact0/100->0/100. TRAIN diagnostic32 WER
41.46%->32.53%, not a claim about full2048. First-slot adjacent repeats23.21%->15.79%,
all-slots-EOS47/100->50/100; full62BPE earlyEOS68/196->70/196. Capacity stop is
not necessarily looping, and naturalEOS is not automatically premature truncation.

QUALITY_STOP unchanged: validation WER/margins pass, allEOS fails; TRAIN32 WER
32.53% remains above20% and20/32 have capacity stops. Native segmented oracle
also fails allEOS, but neither changes the rule nor proves learned text is good.
Fixed8 privileged diagnosis: native source-memory gold-prefix CE0.0434 vs received
memory gold-prefix content CE1.6659/accuracy61.73%; common-available-position
gold/free-prefix CE1.6690/3.6098. Source/native memory/mask not transmitted;
position-aligned diagnosis does not prove exposure bias is the unique cause.

No CLIP/Common/LongCLIP/image/diffusion/FSO participated. Clean40dB is a sentinel,
not measured opticalSNR. Logged GPU-program wall24.48min/training-with-probes18.49min,
peak allocated2.17GiB/reserved2.51GiB, not CPU collection time or a hardware benchmark.
One training run/seed42 and reused qual_dev, not independent-test/generalization
or continuous-vs-digital superiority evidence.

Next recommended experiment NOT_IMPLEMENTED: matched clean CE vs CE+TRAIN-only
frozen native-BART output-distribution distillation; keep initialization, data
order/size, optimizer/LR/updates, unrestricted decoding and symbols matched.
Teacher/original text/length/mask/logits never actual RX inputs. Check gold-prefix
content fidelity and validation recovery, then separately consider SS or training
unlikelihood, not stacked confounded changes. No automatic FSO/joint/image run,
gate relaxation, environment update, data/weight publication or license choice.
User authorizes saving this result/handoff and making this exact repository public;
privacy/publication preflight and visibility verification are separate from model
quality. No task PID is claimed by this documentation-only turn.

## 2026-10-15 V12 matched CE/KD and privileged spatial-semantic receiver probe

User authorizes implementing the two proposed diagnostic experiments, not new
server jobs. Add docci_semantic_probe_v12 only; preserve V6-V11 sources, weights,
data, failed/results and all historical quality gates. Default physical GPU1,
existing offline Python3.8/Torch1.12/Transformers4.30.2/Diffusers0.20.2 only.

Text: approved V11 final20480 weights split into NEW-optimizer CE and CE+full-vocab
TRAIN-only frozen-BART KL,4096 updates each, matched init/order/LR/batch2/TRAIN2048/
qual_dev100/9216complex/R1/unrestricted greedy generation. Teacher/source memory,
gold prefixes/masks/length never actual RX inputs. No FSO/Common/CLIP in text task.
Keep old WER/margin/EOS quality gates; prefix diagnosis uses published FP32
roundoff-bound verifier, not a silently weakened quality threshold.

Image: approved I2 final512 checkpoint and VAE frozen, same399843-parameter
three-scale receiver residual modules. A0 uses image-derived self context only;
A1 adds privileged original-description Long-CLIP vector. BOTH use clean/mask25
alternating TRAIN2048,2048 updates/accum2, identical fixed4096complex/R1 payload.
Shared stored validation observations for original I2, trainedA0 and A1 matched/
shuffled/zero semantics. Full Long-CLIP246-token windows, weighted pooling, no
truncation. Text side info is NOT transmitted/charged: not end-to-end or equal-
total-budget joint communication. Mask25 is synthetic feature erasure, NOT FSO.
PSNR predeclared primary; SSIM and all cases kept; LPIPS NOT_MEASURED, no downloads.
No HiFiC/GAN/diffusion or bidirectional-Common success claim.

CPU contract/report tests and actual historical stored-frame regression executed
locally; Torch/HF unavailable,6 model tests SKIPPED, real saved-backbone/512/GPU
smoke still mandatory. launch requires actual tests -> discarded fresh smoke ->
independent CPU acceptance -> NEW main process from approved parent, finite cap,
no auto retry/promotion/other-process stops. Final CPU collection separate, original
run read-only, no legacy180s report cutoff. README has exact upload/install/GPU1/
run/progress/collect commands, forecast1-3h NOT measured, cap6h not ETA. New source
tar/SHA and atomic GitHub synchronization required. No server task PID to report.
