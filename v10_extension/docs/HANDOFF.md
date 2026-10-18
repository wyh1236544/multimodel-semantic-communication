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

## 2026-10-15 V12 HTML collection JSON import repair (fix1)

User returned a collect.py traceback: original report.render uses json.dumps but
never imports json. Failure occurs after original verify and record copying,
while writing the derived HTML, before archive creation. It does not require
GPU training again and is not evidence of model quality. Original test_render
used an empty task map, so did not exercise either failing text/image branch.

Add [docci_semantic_probe_v12_collect_fix1](../docci_semantic_probe_v12_collect_fix1/README.md).
The isolated CPU entry supplies stdlib json only in the original renderer's
runtime globals, delegates unchanged verify/collect and restores globals on exit.
No original V12 file, manifest, source snapshot, checkpoint, result or quality
gate changes. All source/record hashes and failure propagation remain enabled.
New output records retain repair source and COLLECTOR_FIX1.json provenance.

Six new CPU synthetic methods (six main/smoke task combinations, real legacy tar
and renderer, escaping, source preservation, conflict and error paths) plus all
16 original CPU tests passed. No Torch, pickle, GPU, environment changes or new
server jobs. Actual user V12 record collection remains to be performed on server.
Preserve failed .contents and use NEW *_records_collect02.tar.gz outside the run;
if occupied use collect03. Do not edit old report.py or bypass hash verification.
Source-only archive 6777 bytes, SHA256
c1f42ed841e3a971248fde86dd56ef51302e43b328ec73526d27f4485a21801a.
Prior shared handoff backed up byte-for-byte under deployment_records before append.

## 2026-10-15 V12 actual records verified; V13 offline semantic evaluation

This entry supersedes "V12 not run / recollection pending" as CURRENT state only.
User supplied collect02 SHA256 434d8e43521c9fa75d16b6f0e0d968daf566172ab02b392e4ca374eb0e614683.
735 original light-file hashes and trusted CPU verification previously passed;
no uploaded code executed, no checkpoint/pickle loading or new GPU inference.
Canonical aggregate-only public results: [V12 results](results/v12_semantic_probe_20261015/README.md).
Same100 reused qual_dev, seed42: V11 WER46.2206%, V12 CE45.2719%, KD44.7521%.
KD-minus-CE -0.5198pp, conditional paired95%CI[-1.0956,+0.0637]pp crosses0.
Both remain QUALITY_STOP and exact0/100. No automatic metric or gate relaxation.
A0 image-only receiver adaptation improves synthetic mask25 PSNR0.541243dB,
clean slightly worse. Correct-vs-shuffled uncharged oracle text effectively zero:
not content-specific mutual reconstruction, not equal-total-resource joint communication.
No FSO, diffusion, LPIPS or digital baseline in V12.

User now authorizes new offline evaluation code and GitHub sync of code/current aggregates.
Add [docci_semantic_eval_v13](../docci_semantic_eval_v13/README.md) only; old training sources,
weights/results/manifests immutable. Shared docs backed up before replacement/append.
V13 is an evaluation version, NOT a new trained model. Protocol frozen before new
semantic scores but AFTER inspecting historical WER: retrospective exploratory analysis.
Read-only JSONL/tar adapters independently recalculate WER; reject incomplete matrix,
wrong split, changed reference hashes, truncation, changed clean9216complex/R1 contract.
Real100-description/1100-output preparation passed; source-only stratified30 audit
descriptions cover330 candidate judgments, not just30 candidate outputs.
Human source facts frozen first; identity-masked candidate judgments then evaluated;
unreviewed/missing matrices fail, no invented SPICE/LLM fact scores.
Primary facts retain unclear denominator; source-unsupported is NOT visually false.
Independent optional CPU Longformer4096/layer7 BERTScore0.3.13; full-text tokenizer
preflight, overflow abort, no silent truncation/drop or CLIP/BART self-evaluation.
No verified model-specific rescale table bundled, raw P/R/F1 with empirical derangement
reference; no invented baseline or score-as-accuracy claim. Native BART is privileged
diagnostic, not theoretical upper bound; actual RX shuffled differs from deranged descriptions.
Digital NOT_RUN, clean40dB sentinel, graceful degradation is untested hypothesis.

14 new synthetic CPU tests including optional scorer mocked entry;16 original V12
CPU regression tests passed. Python3.8 parse, per-file archive integrity and isolated
restore tests passed. Real pretrained evaluator download/inference NOT_EXECUTED;
manual fact inventory/judgments PENDING. No changes to server Conda/global settings.
No training/server process launched or PID owned. Optional dependencies only in a
separate user-created LOCAL virtualenv, never install into existing server environment.
Source-only release v10_docci_semantic_eval_v13.tar.gz,22798bytes,SHA256
d803aa7e164de1c94d21cd60caba1064be8be3b5fbe534780b64c94bd5c2d0fb.
Direct additive tar layout v10_extension/docci_semantic_eval_v13, no install.py.
Upload/unpack/check/manual-audit/optional-model commands in README. Do not rerun
training merely to evaluate saved text. No raw DOCCI text/images/records/weights
or private full-text review HTML in public Git; publish source and aggregate-only charts.

## 2026-10-15 V13 server CPU overlay import passed; model download still pending

This entry supersedes the earlier LOCAL-only deployment recommendation, not any
frozen scoring code/protocol, quality gate or measured result. User's Mac RAM is
limited; school server reported125GiB total/86GiB available RAM. User authorizes
offline CPU evaluation there using a project-only venv with system-site-packages,
not an upgrade to the original Conda. No new training, GPU job or global setting.

User terminal confirms sys.prefix!=sys.base_prefix and imports torch1.12.0,
transformers4.30.2 and bert-score0.3.13 in the new overlay. This is USER_REPORTED
import evidence, not an independently executed server check or real forward.
The only added wheel is official universal bert_score0.3.13,61135bytes,SHA256
bbbb4c7fcdaa46d7681aff49f37f96faa09ed74e1b150e659bdc6b58a66989b9.
No reinstall/recreation is needed; preserve that working overlay.

Frozen prepared inputs independently reverified locally:100 descriptions/1100
outputs, status PREPARED_LEXICAL_ONLY. Original V13 source/MANIFEST/source tar
and prepared bytes unchanged. Fixed independent Longformer layer7/CPU/batch1,
full-text no-truncation policy and raw P/R/F1 remain; manual facts PENDING,
digital NOT_RUN and historical QUALITY_STOP unchanged.

Official allenai/longformer-base-4096 revision301e6a42cb0d9976a6d6a26a079fef81c18aa895
is still DOWNLOADING locally at this entry. Weight expected597257159bytes/SHA256
06c56757f0510de87c231acd03650ca204c5ed65c281cd1081b98288183f2468.
Do not infer completion from this entry or partial files. Private local download
state has exact session/PID/range paths; audit current process and output before
resuming. Full offline materials archive NOT_BUILT;10-output CPU smoke and full
1100-output scoring NOT_EXECUTED. Never supply commands claiming the full kit
exists until full hash verification, packaging and member checks actually pass.

Next: verify model bytes -> privately package model+prepared inputs+wheel ->
server checksum/unpack -> frozen-input verify ->10-output CPU smoke in NEW output.
Only then consider full scoring. No automatic retry or background notification
promised. Raw descriptions/private HTML/model weights must never enter public Git.
This documentation-only update is backed up under deployment_records/v13_server_cpu_20261015.
See [server CPU deployment status](V13_SERVER_CPU_EVAL.md).

## 2026-10-15 V13 evaluator download and private offline runtime assets verified

This supersedes the previous entry's download/package-pending status only.
Initial transfer exited on ChunkedEncodingError at87.06%; resume02 exited on
three part3 TLS failures at94.52%. All original bytes/partials preserved.
Resume03 independently completed parts1/3; failed parts0/4 were confirmed STOPPED
before single persistent-session resume04 appended to those exact two files.
TLS verification always enabled; no mirror, certificate bypass or environment
change. Transient SSL errors observed; unique network/proxy/server cause NOT proven.
Entire597257159-byte official weight SHA256 verified against LFS value
06c56757f0510de87c231acd03650ca204c5ed65c281cd1081b98288183f2468.
Small-file pinned revision and all member SHA256 values checked, no model loading.
Resume download processes completed/exited; no school-server process launched.

Private runtime archive v13_server_cpu_offline_01.tar.gz,377419041bytes,SHA256
e8cbcfcd8fc02cccdf1bd5bd60caa57c2e2ae7eae6e95a4caf392e55d93f7b36.
18 regular USTAR members; streamed per-member hashes independently matched source,
manifest and SHA256SUMS. Contains official model/download receipt, original immutable
100-description/1100-output prepared inputs, tiny universalwheel and deploymentREADME.
No caches, incomplete ranges, Mac environment, project weights or image dataset.
Archive contains raw descriptions/private HTML and third-party model: PRIVATE delivery
to user's server only, never public GitHub. This is runtime material, NOT new code.
Frozen V13 source tar remains22798bytes/SHA256
d803aa7e164de1c94d21cd60caba1064be8be3b5fbe534780b64c94bd5c2d0fb.
Original source/prepared hashes unchanged; no training/checkpoint/gate modifications.

Use already-working server CPU overlay, upload archive+checksum, new directory extract,
member checks, frozen-input verify, then NEW10-output CPU smoke. Real pretrained model
inference NOT_EXECUTED locally or on server; new BERTScore NOT_MEASURED, human facts
PENDING, digital NOT_RUN, historical QUALITY_STOP unchanged. Do not treat download or
packaging verification as semantic-quality evidence. No automatic full evaluation.
Local verification receipt and exact operational PID/session/failure logs retained under
releases/; no credentials/raw descriptions printed into public deployment docs.
Previous GitHub write attempts timed out during approval, no commit/ref update;
current local deployment-doc updates remain UNSYNCED until a later successful save.


## V14 clean single-modality baseline calibration (2026-10-15)

User authorizes implementation and source synchronization, not a remotely started GPU job.
Add docci_baseline_v14; preserve V6—V13, historical checkpoints/results and quality gates.
Offline physical GPU1, existing Python3.8/Torch1.12/Transformers4.30.2; FP32, no installs.

Readonly calibration evaluates approved actual V12 B1 text and approved I2, plus
same-input native BART/VAE privileged references. Clean SNR=null; I2 internal40
sentinel is not measured SNR. No FSO/Common/CLIP/diffusion or digital comparison.

Fresh text legacy64 vs released-code-aligned64 share data/CE/presentation budgets,
32x64 per chunk, all9 frames/9216complex/R1, no RX source length/mask. Paper's
Table13 hidden768 contradicts released default narrow64; optional wide768 is a
capacity hypothesis, NOT official default or a claimed full paper replication.
Unrestricted greedy and deployment beam4/repeat control are separate for all arms.

Image data64 vs expanded start from SAME approved I2 tensors, same updates/LR/
pixel_MSE+.01latent_MSE/4096complex; VAE frozen but differentiable at input.
Pilot2048 and explicit full9647 supported; CPU disk-backed bounded LRU, no truncation.
No automatic promotion or smoke weights reuse; frozen/checkpoint/source audits.

Independent CPU verifier checks matrices, WER/frames, source and actual training
presentations. New private light archives exclude weight/cache arrays; retain actual
text/PNG and honest channel/LPIPS status. Optional explicit offline LPIPS assets,
no automatic downloads. Human facts source-first, blinded, no synthetic fact scores.

Local stdlib/NumPy/integrity tests are separate from mandatory real Torch/HF tiny
contracts and authentic GPU/backbone smoke. Local Torch unavailable; numerical
and recovery results NOT_EXECUTED. No owned school-server PID or training initiated.
Source USTAR/checksum, isolated restore and atomic GitHub source save required.
Initial recommendation: launch.py calibrate NEW --task both, then separate text
pilot, not simultaneous text/image/full/FSO. See README for full commands/limits.

## 2026-10-15 V13 actual CPU scores returned

The previous download/score-pending entries are historical only. User returned1100
actual independent Longformer P/R/F1 rows. All case IDs/counts/finite scalar values
and11 group means were independently rechecked; no local neural inference rerun.
V12 KD matched rawF1=.927635, shuffled=.844556; not92.76% accuracy. Manual facts
PENDING, digital NOT_RUN, original QUALITY_STOP unchanged. Aggregate-only public
record: results/v13_bertscore_20261015/README.md. No raw descriptions/images/weights
published. New V14 training has NOT_EXECUTED server status; no job/PID owned here.

## 2026-10-16 V15 implementation and returned V14 calibration/image results

This supersedes prior V14-not-run status ONLY for actual calibration and image
pilot02. Text legacy64/released64 new training completion is still unconfirmed.
Canonical aggregate evidence: [V14 returned results](results/v14_baseline_20261016/README.md).
Same reused100 qual_dev, clean/SNR=null: oldI2 20.0913dB, data64 22.9772,
expanded2048 23.4150; two new arms same5120updates and parent I2. NativeVAE26.2312
is privileged, not equal resource. No FSO/Common/diffusion; seed uncertainty not measured.
Prior pilot01 reserve-guard failure preserved, not proven OOM/other-user cause.

User approves the staged plan. Add [V15 read-only semantic evaluation](../docci_semantic_eval_v15/README.md),
20 synthetic CPU contracts plus14 V13 regressions, source USTAR16-member and
isolated restore tests passed. Real1100 prepared/BERT scalar rows rechecked;
actual V14 calibration adapter1900 combined text outputs passed. No neural
inference locally, no new training, no school-server PID or process started.
Full scope/boundary: [V15 handoff entry](../docci_semantic_eval_v15/SHARED_HANDOFF_ENTRY.md).

Human source-first30/330 audit PENDING;15development/15confirmation are evaluator
subsets within REUSED qual_dev, not new model test. Forty hand-authored synthetic
challenges generated, actual BERT challenge inference NOT_RUN. CAPTURE/AlignScore
have frozen scalar import contracts ONLY; actual runtimes not implemented/tested.
Raw similarity and anchor position NOT accuracy. Original QUALITY_STOP unchanged,
digital NOT_RUN; no continuous superiority or joint mutual recovery established.

Source archive34660bytes/SHA25608fa008ff0aad5b041059e322a95478b4b424e4efef1237bfdc7a20e878de748.
Code/aggregate docs synchronized via GitHub tree/commit and expected-head lease,
not local git (this mirror is not a checkout); no force push or old file deletion.
Source/private result packages are distinct. Publish NO raw DOCCI text/images,
private recovery HTML, complete records or weights. Old shared docs byte-backed-up
under deployment_records/v15_before_20261016. Current next commands in V15 RUN_NEXT:
read-only server task audit, then V14 text pilot on physicalGPU1 ONLY if not already
running; preserve failures and >=4GiB runtime free reserve. Future full-data/LoRA/
InfoNCE/framing/noisy/digital/joint experiments NOT implemented in this release.

## 2026-10-16 V15 existing V14 task guard: no duplicate training

User reports V14 text pilot01 already running and explicitly requests adapting
previous code so it does not duplicate that task. Do NOT restart or create a
second text pilot. Actual PID/completion not observed locally. Add only
docci_semantic_eval_v15_guard; all original V14/V15 source+manifest byte hashes
verified unchanged. Current-run instructions supersede old V15 RUN_NEXT startup
example; user keeps current GPU1 training uninterrupted.

guard.py status/prepare defaults to exact existing experiment_results/v14_text_pilot_01.
No optimizer, Torch/GPU, model inference, subprocess/job creation, downloads or
automatic polling. NOT_READY exit2 creates no output; FAILED exit3 preserves;
outer LAUNCH_FINISHED+main/source/task identity and original CPU verifier receipt
required before importing. Does not rerun full V14 frame/history verification.
V15 still verifies light hashes/text matrix/WER. One deterministic output per
run, atomic mkdir claims; partial/concurrent exit4 never auto clears/retries.
Same completed identity reuses existing validated prepared output without
re-preparation or BERTScore inference. New V14 BERTScore NOT_MEASURED, factsPENDING,
historical quality unchanged. No new recovery/continuous/joint claims.

New source pack includes byte-identical original V15+guard and a preflighted
additive installer: absent V15 added, matching existing files unchanged, unknown
conflicts rejected BEFORE writes. Payload allowlist excludes all V14/run/weight
paths. 17 new synthetic contracts+20 oldV15 regressions passed;24 regular USTAR
members/Python3.8 parse/isolated install+repeat-zero-write/status+prepare NOT_READY
no-output tests passed. Synthetic ready/import fixtures are NOT actual current
server text results; real completion integration remains unverified.

Archive49214bytes/SHA256528681999a9ab0a4698847511b8d3c4b69f775deb4a5b043950488abd22a71cf.
Direct wrapper _v10_docci_semantic_eval_v15_guard/install.py --project-root .;
do not just look for installed guard.py before running installer. Exact commands,
exit-state interpretation and original V14 collection in module README. Old
source/release/results untouched; no raw data/weights published. Shared docs
backed up under deployment_records/v15_guard_before_20261016. Save source and
shared current-status docs atomically to GitHub with expected-head lease; no
force push. No server process started or other-user process touched.

## 2026-10-16 V14 text completion confirmed; protected RX bridge V16 implemented

Current evidence supersedes earlier "V14 text still running/not returned" only.
Trusted private V14 text archive SHA43f5908558a64e558937c53b39cc3eed4a94f2ba4ad2b663cd81307dfe5c9760
passed317-light-file/1496-row independent CPU/source/training/frame checks.
Same2048TRAIN/20epochs/5120updates/effective8: legacy64 greedyWER55.6877%,
released64 36.9953%, deployment34.2017%; clean/SNR=null. Correct/real wrong/zero
released greedy37.00/111.43/388.49%; empty606/606 zero content, full62 earlyEOS25/196.
Repetition and fact errors remain; no full exact descriptions. Same100 reused
qual_dev, one training replicate; no FSO/CLIP/Common/diffusion/joint gain.
Aggregate-only evidence: results/v14_text_confirmed_20261016/README.md.
V15 import privately verified100sources/2500outputs; new V14 BERTScoreNOT_MEASURED,
factsPENDING, CAPTURE/AlignScore/digitalNOT_RUN. No raw data or weights published.

User now authorizes implementation of the reviewed revised architecture, not a
remotely started job. Add ONLY docci_rx_bridge_v16 plus backed-up shared docs.
Pin actual released64 SHA3e6743eb1bcdbe4bed5db14ba9c9d5cb6d30cb2676066fc41c63fbb09ee3b000
and expanded-I2 SHA95c05e3b374deae2a3bd10603b8e95ebbd464456f7581a054f2b80597be5923b.
Freeze both codecs/BART/VAE; preserve all historical sources/manifests/results.
Exact architecture count6,916,866 trainable bridge parameters,
independently checked by formula/runtime; no full backbone unfreezing.
Fixed18432text+8192image reals=13312complex count/R1, no new Common or extra text.

First image→text only: self adapter, privileged source-image oracle Long-CLIP,
actual reconstructed-image Long-CLIP and actual unfused I0 spatial features.
Equal196x768 support/64x256 query bridge, gated new cross-attention AFTER existing
BART text cross-attention norm and BEFORE FFN in last2layers. Original memory,
weights and HF generation-cache format unchanged. Gate-off must reproduce actual
baseline tokens; real tiny tests cover causality/incremental cache/beam expansion.
Content gate reads first FREE EOS probability after[decoder_start2,BOS0], never
source length/mask/true empty-slot labels. Preserve real donor/zero/off controls.
Long-CLIP extra compute explicitly differs from I0/self; not equal full-system FLOPs.

Cache frozen features for all9647 officialTRAIN+100reusedqual_dev; no test inputs.
Real batch64 contrastive alignment, CE micro1/accum8, TRAIN-only2alignment epochs
and3CEepochs, fixed last checkpoints. Same budgets/data/order/init across arms.
Oracle and actual trained separately, no unlabelled input-distribution switch.
Launcher stages: check/smoke → cache → explicit oracle/self → explicit actual.
No automatic promotion or duplicate self/V14 training; oracle numerical screen
recomputed from sealed rows before actual. Failure stops this connector/config,
not all multimodal recovery. Full semantic success remains false until independent
fact improvement; new facts/BERTScore unmeasured. No FSO/digital/LPIPS/diffusion.

13 new local stdlib/NumPy CPU/storage/collection contracts and32 oldV14 CPU
regressions passed. Seven Torch/HF contracts explicitly SKIPPED locally (absent
dependencies); required on server, followed by authentic8TRAIN/2VAL saved-weight
smoke for all4arms. Real BART/VAE/CLIP/GPU, runtime peak/ETA and effectiveness
NOT_EXECUTED. No school-server PID owned. Source USTAR, checksum, independent
restore/conflict tests and atomic GitHub save required. No environment installs.
DefaultphysicalGPU1, startup>=16GiBfree/runtime>=4GiBreserve; not measured peak.
Main cache min64GiBdisk, about23GB dense tensor payload before compression.
Collector independently hashes large arrays/checkpoints on server, exports PRIVATE
light records with actual text/images/HTML, excludes arrays/weights. CPU cap is
finite but not old180seconds. Local light replay must say arrays=False, not claim
new neural inference. Shared pre-edit bytes saved in deployment_records/v16_before_20261016_01.

Current next user action: upload source archive/checksum, unpack NEW module,
launch.py check then ONLY launch.py smoke in NEW output. Return collected smoke
records before full stages. README contains all exact commands and honest timing
limits. Do not auto start training, stop other jobs, or publish DOCCI recovery data.

## 2026-10-16 V16 missing source_tokens CPU verifier compatibility fix1

User returned report.py/index_check KeyError('source_tokens') after the cache
stage. Original cache.create stores source_tokens in index but omits it from
baseline.jsonl; the original synthetic fixture included it and missed the bug.
This is a confirmed producer/verifier schema mismatch, not a model-training
or recovery-quality conclusion. Actual source cache integrity still requires
the repaired independent check on server; no server process started here.

Add docci_rx_bridge_v16_fix1 only. Original V16 source/MANIFEST, snapshots,
raw baseline/index/arrays, model settings/predictions/thresholds remain immutable.
Original V16 MANIFEST SHA2567ea2874353f748c4f7f9bbae8d95ef6475def1804e388d505976bcb1732c89eb.
Runtime CPU report compatibility supplies ONLY absent baseline source_tokens from
the sealed index after checking source ID/text/SHA/split. arrays=True keeps the
original9x64 ID/count verification first; present wrong/null values and corrupted
index/arrays still fail. Never infer source length from generated text and never
pass source lengths to receiver forward. Original context restored on exit.

Report adds only a NEW derived VERIFIED.json; differing existing receipts are
preserved/rejected. Resume-smoke reuses a completed original smoke_cache and runs
only the not-yet-started4-arm short training/evaluation in NEW output. Refuse if
original smoke_train already exists. Original failed launch and raw cache remain
preserved. No cache regeneration, V14 retraining, repeated completed training,
automatic full-stage promotion, environment changes or other-process actions.
Future run/report/collect go through fix1 entries; old source is not hand-edited.
New run/light archives include explicit compatibility source snapshots/provenance.

11 local CPU regressions passed: real original producer expression reproduces
the KeyError, fixed independent verification/collection, immutable original bytes,
source/count/array tamper refusal, receipt conflicts and context cleanup, unchanged
effective configuration, real CPU child verification in mocked-GPU orchestration,
and duplicate-training refusal.13 original V16 CPU regressions passed. Mocked
GPU child copies TEST_ONLY storage fixtures, NOT neural/GPU effectiveness evidence.
Python3.8 parse/source USTAR/independent restore executed. Authentic server/GPU
resumption remains unexecuted locally. Return private fix1 smoke records.
Shared docs backed up under deployment_records/v16_fix1_before_20261016_01.

## 2026-10-16 V16 formal evidence returned; V17 bounded overnight bridge implemented

V16 no longer unexecuted: cache archive SHA3eaa7074695c50fca6472f96e4ec0dfee684f1d3787762d953f7d9a9fba3bf5f
and oracle archive SHAe35d859592e221fc803d38f9ddb84c771b805666c0fe1bf73c355866aa4c03d5
passed independent trusted light verification (95/207 file hashes;864 generated rows;
same approved cache). Local arrays=False/no Torch/no neural inference. Server receipts
cover excluded arrays/checkpoints. Aggregate-only evidence in results/v16_oracle_20261016.
Clean100qual_dev WER original36.9953/self36.6213/oracle35.9405/wrong35.5306/zero36.7094%.
Oracle−self gain.6808pp CI[−.0885,1.4862], correct−wrong gain−.4099pp CI[−.7188,−.1145].
Formal302alignment/3618CE perarm,9647TRAIN, frozen parents/learning intact, numerical
screenFAILED/factsPENDING. Do not run V16actual or silently lower its thresholds.

User explicitly authorizes larger overnight V17 implementation after plan review.
Only add docci_segment_bridge_v17; preserve old V14/V16/Fix1 sources/manifests,
weights/results/cache. Shared docs pre-edit backed up under
v10_extension/deployment_records/v17_before_20261016_01 with checked hashes.
New segment-query residual receiver memory repair then unmodified frozen BART;
formula4,429,826 bridge parameters (actual Torch count still required).
Native frozen BART TRAIN-only content contextual means replace random teacher
projection; image/text contrastive alignment, segment cosine supervision, pair
matching BCE and main CE. No wrong-caption CE maximization, no inference source
text/masks/lengths. No new pretrained downloads, full unfreezing, Common or diffusion.

Four equal-parameter arms: A/B self/oracle clean; C/D self/oracle mixed clean/20/10/5/0dB.
PhysicalGPU1; TX-wire AWGN followed by original codec and received-memory EOS gate;
never reuse clean gate. Image ideal/oracle, not real equal-resource/FSO system.
Fixed13312complex-symbol count/R1. TRAIN9647/VAL100 reusedqual_dev, no test. Two real
batch64 alignment epochs302updates and six CE epochs7236updates/micro1accum8 perarm.
PriorityC,D,A,B fixed before runs. PrimaryDvsC awgn0; secondaryBvsA clean. Evaluate
all100 at clean/10/0, matched/5real donors/zero/off, donor means within description.
Off explicitly reuses independently generated frozen baseline, identity repair
checked; no claim every arm regenerated off neural outputs. Facts remainPENDING.

One bounded launcher: tests→read-only full-cache audit→actual saved-weight4arm small
smoke→native TRAIN teachers/VAL observations→C/D/A/B→independent verification→PRIVATE
auto collect. Eight-hour hard budget including collection reserve15min; estimates
unmeasured, new memory repair backprop differs from V16. Numerical/engineering/
resource failures stop, no auto retry/epochs change/other-user kill. Per-epoch saved
material is not exact resume. COMPLETE only all4main+smoke verified; partial/notrun
retained and paired results only for genuinely completed pairs. No autoactual/FSO.

16 local CPU protocol/storage/owned-process tests passed;8Torch/HF contracts
explicitlySKIPPED absent local dependencies. Server requires them plus actual
BART/V14/GPU smoke; new performance/peak/ETA NOT_EXECUTED. Source Python3.8 parsing,
USTAR independent restore/conflict checks and GitHub atomic save are delivery
requirements. No server job started here; raw DOCCI/records/weights never public.

## 2026-10-17 V17 returned; V18 original text-codec noise adaptation

Latest evidence: docs/results/v17_text_bridge_20261017/README.md. PrivatearchiveSHA
5cb0e40eec4e594bd259f7703d869e5329b584deccfeae3dd4e0052808f0f121 passed1348filehashes,
200trustedsourcecomparisons and7200mainrows. LocalarraysFalse/noTorch/noinference.
PARTIAL_BUDGET: C/D/A complete, B154/302alignment only, noCE/eval. At0dB
A/C/D WER85.8277/84.0602/84.3691%. CvsA+1.7675pp; DvsC−.3089pp
CI[−.7470,.1341], matchedvs5donors+.2992pp belowpredeclared1pp.
Drepeat34.1830% vsC32.9989%, emptyfalse19/606. No visualextrasuccess,
noFSO/digital/actualRXimage. Frozenoriginalcodec/BART, factsauditPENDING.

User authorizes nextcode, not serverexecution. Add only docci_text_channel_v18
stableconfigrunner; oldV14/V16/V17/source/manifests/data/weights/results unchanged.
Noarchitecturalparameterincrease: original25618432 TXcompressor+RXreconstructor
train, BARTfrozen. NativeBARTsourcefeatures reused from v14_text_pilot_01/main/text,
never cachedfrozenTXwire. ParentcheckpointSHA3e6743eb1bcdbe4bed5db14ba9c9d5cb6d30cb2676066fc41c63fbb09ee3b000.
Twoarms sameinitial/schedule/2048TRAIN/10epochs/2560updates/micro1accum8,
twoseparateforwardbackwardviews/description. clean_continue has twocleanviews;
noise_replay25%clean+75%random0/5/10/15/20dB. Newoptimizer,fixedlast,noexactresume.

Originalfrozen,clean_continue,noise_replay each2000freegenerations: clean100;
5noiseSNRs*100*3draws; shuffled/zero atclean/0dBr0. Repeatsaveragedwithindescription
before10000pairedbootstrap, seed42, pointwiseunadjusted, reusedqual_dev, notfinaltest.
Primary0dBnoisevs clean; operationscreen>=2pp+CIpositive andcleanCIgainlower>=−1pp;
stillnotfactualsuccess. NoCLIP/Common/imageupdate/FSO/digital/noenvironmentchanges.

Pipeline tests->actual8TRAIN/2VALsmoke->read-onlyfullinputaudit->bothtrainarms->
threeevalpaths->independentCPUverify->PRIVATElightcollection. Eight-hourcapincluding
10mincollectionreserve. Rough3–5h estimate fromoldV14speed only, newGPU/ETAunmeasured.
GPU1start16GiBreserve4GiB, nootherjobcontrol/noauto retries/cache regeneration.
12CPUcontracts+32oldV14CPUregressions passed;5Torch/HFcontracts explicitlyskipped
locally, mandatoryservergate. Realweights/GPU/quality remainNOT_EXECUTED.
Before-editsharedbytes retained in deployment_records/v18_before_20261017_01.
Deliver sourceUSTAR+SHA/servercommands andatomicGitHubsource/aggregates only.

## 2026-10-17 V18 actual evidence; V19 matched practical digital and OOD inference

V18 is actually COMPLETE, not still waiting to run. Public aggregate:
results/v18_text_noise_20261017/README.md. ArchiveSHA
bec504da2963dd143880e46c3a51c6e816459b65883bf082077a238671d23791,
217filehashes/17trustedsource/70dependencies/6000main+36smoke rows independently
recomputed locally without Torch or pickle/large arrays. At0dB clean/noise WER
87.1430/72.0354%, pairedgain15.1076pp CI[14.3676,15.8733];5dB17.8566pp.
Predeclared numerical screen passed, factsPENDING/fullsemanticfalse. Repeat remains
15.45%, not gone; emptyfalse1/1818. SimultaneousTX/RX adaptation, no TX-only ablation.

User explicitly authorizes next code, not remote execution. Add ONLY
docci_digital_compare_v19 and backed-up shared docs; old source/manifests/models/
data/results preserved. Shared pre-edit bytes and SHA manifest saved under
deployment_records/v19_before_20261017_01. No school-server process started here.

No new training or architecture. CPU actual raw/zlib/bz2/lzma shortest-message
source coding, protected18byte length/codec header+CRC, N648/Z27 QC-LDPC R1/2
andR3/4, self-written NumPy normalized-min-sum decoder40iterations alpha.8,
unit-energyQPSK. Header2copies, bodylength-only cyclic rate matching,64pilots
estimateN0, all9216complex symbols charged. RX gets no TX/source length or truth,
noARQ. Whole-paragraph CRC/format/decompression failure yields emptytext and
WER1 for nonempty source, not discarded. Conditional postFEC BER unknown when
header cannot be decoded; rawbit counts and denominators retained. Not fullWLAN
PHY/optimalMCS/capacityachieving claim. Standard numeric tables cited inprotocol.

Local actual6800digital+12smoke packets and full deterministic receiver replay
completed. Public aggregate results/v19_digital_cpu_20261017/README.md.
R1/2success clean100/100,0dB298/300,5–20each300/300;0dBWER.6667% vsV18noise72.0354%.
R3/4success0dB277/300, alsofullyreported. R1/2 at−2/−4/−5dB244/58/1 of300;
−6/−8/−10zero. VerylowSNR header failure dominates; no inference that continuous
wins there. Asymptotic capacity length-fit references separate, not fakeWER0.
No LMcompressor/progressive packetized digital, factsPENDING. One seed,reused100
qual_dev,3noisedraws averagedwithin description, bootstrap10000 pointwise CIs.

V19 preserves original frozen/clean/noise 0–20dB saved outputs. Pending GPU-only
OOD evaluation−10/−8/−6/−5/−4/−2dB3draws, three pinned checkpoints;12actualclean/0
anchor rows and6negative smoke rows first, no training. Approved V18 clean SHA
4a0a97034e7b7c2ff48303bfbeadd31677542cccf95ee9ad5e9ba8b02e83f6b1,
noise96b27347b62c918a931ba221640139ce3e162b8ff8fbcaebde89f6b13cc5312b.
V18 schema differs from old V14; load exact approved state only after SHA check.
Default oldrun experiment_results/v18_text_noise_01; needs actual server weights/
V14nativefeatures, not just a light records archive. NoGPU OOD run occurred locally.

20newCPUcontracts+12V18CPU+32V14CPU passed;5Torch/HFcontracts explicitlySKIP locally,
mandatory on server before authentic saved-weight smoke. Source Python3.8/USTAR/
isolatedrestore and atomicGitHub save required; no dependency installs. PhysicalGPU1
start16GiB/runtime4GiB safety floors,4hour example including5mincollect reserve,
no auto retry/jobkill. CPU-onlydigital first possible; laternegative uses NEWroot
and readonly completed digital root, not duplicate main experiments. Independent
collection may CPU-replay received packets as verification. Preserve human fact CSV.
Source-only package+SHA and run commands inV19README; rawtext/PRIVATEHTML/records/
weights never public. Jointbridge/image/FSO/lowbandwidthmodel training stay NOT_RUN.

## 2026-10-17 V19 returned COMPLETE; image-led text guidance V20 authorized

V19 returned archive SHA112baa834c9af6ee2c919c1c186be4dda369d90a54bd31098d166eefe82b6a5c,
147 regular members/146 hashes/18 trusted source files/87 dependency snapshots.
6800 digital,5400 negative inference,4800 reused rows,12 real generation anchors
verified. No new training. All original bytes preserved. Cross-platform exact
float replay failed at ~1e-15 telemetry/1ULP capacity; separate read-only audit
keeps exact captions/CRC/bit counts/decisions,rel/abs1e-12 telemetry,8ULP capacity
(actual1ULP), and reproduces ALL REPORT numerics/CIs exactly. No original verifier,
threshold,source,manifest or outputs changed. Local no Torch/pickle/neural replay.
Public aggregate results/v19_returned_20261017, no private descriptions.

At−5dB V18noise WER89.7928 vs digital1/2 99.6667, gain9.8739pp
pointwiseCI[8.9033,10.7158]; repeat40.79%. At0+ digital far better.
At−4 1/2 still better mean,−10 gain CI spanszero. FactsPENDING.
Do not request repeat V19 or claim useful semantic restoration from lowSNR WER.

User explicitly reprioritizes image-text aid over word-perfect text, approves
coding and longer training. New module ONLY docci_text_guided_image_v20; old
source/manifests/data/cache/weights/results preserved. Shared doc pre-edit exact
copies in deployment_records/v20_before_20261017_01. No live server access/job.

V20 first direction text→image: V14 expanded image SHA95c05e...5923b,
V18noise text SHA96b273...5312b. Image encoder/decoder and four128-wide spatial
cross-attention/resampling modules train; BART/VAE/LongCLIP/V18codec frozen.
Three equal-parameter arms A_self/B_tx_text/C_tx_rx_text. No matching gate.
Source LongCLIP full descriptions246-content chunks then32slots; received
9x32x768 memory from actually-noised V18 wire. NativeBART memory ONLY labelled
oracle evaluation probe, never actual-RX training. TX cannot see receiver oracle.
No new Common, diffusion, FSO, reverse image→text or image digital benchmark yet.

Audit known V14 image/text TRAIN2048 cohort metadata with pinned SHA, actual
DOCCI image bytes and text hashes. Fixed500 known-parent-unseen TRAIN candidates,
all excluded newtrain,first100 main dev;8192 TRAIN/8epochs/16384updates perarm,
micro1accum4. qual_dev100 repeated exploratory only. Old V16/V17 bridges used
allTRAIN but NOT reused; no universal historical/pretraining-unseen claim.
Fixed-final checkpoint, independent epoch backups; no exact auto-resume.

Mixed25%expected clean plus0/5/10/15/20dB;both modalitiesAWGN, same random stream
acrossarms. Image8192+text18432 reals/13312complex payload,64complex pilots charged,
total13376uses,R1. Threeevaluation draws are separateR1 trials, not combining
three transmissions. Clean=null,40 only conditioning sentinel, pilot SNR only atRX.
Loss sameallarms pixelMSE+.01latentMSE+.05officialLPIPS. No CLIP self-judge loss.

CPU independent matrix/source/order/loss/telemetry/PSNR and saved-weight/cache SHA
checks, C correct/donor/zero/oracle keep ownimageTX/noise EXACT same.
Primarynewdev0dB LPIPS CvsB,alsoCvsA/contentdonor;PSNR noninferiority−.2dB.
10000imagepaired bootstrap,noiseaveragedwithinimage,pointwise/singleseed,
no fullsemantic claim without independent facts. Textactual outputs saved onceC,
explicitlynotimage-assisted. PRIVATE actual PNG/text gallery, no bestcase selection.

17newCPU+44oldCPU tests passed;6Torch/HF contracts explicitly SKIP locally.
Actual saved-backbone/real LPIPS/GPU smoke mandatory, no local performance claim.
Official AlexNet233MiB andLPIPS6KB downloaded viafixedofficialURLs/fullSHA,
Torch-only formula avoids installinglpips/torchvision or modifyingserverenv.
Separate offline asset tar, never public GitHub weight upload.
GPU1 start20GiB/runtime4GiB floors,disk64GiB. Pipeline max48h including30min
collectionreserve. Rough20–40h only extrapolation, newarchitectureETAunmeasured.
StageB/C/A main priority,smallall3smokefirst. Ownedchildrenonlystoponfailure/timeout,
noautomaticretry/batchchange/otherjobkill. Independent PRIVATE light collection,
largecache/epoch/finalweights stayserver. Partial evidence explicitly labelled.

## 2026-10-18 V20 first CUDA failure; deterministic pooling Fix1

PRIVATE returned archive SHA0f2fa55db55040375eb671b5ee8863499db507f87562f8b03201b0311f51fcae.
146regularmembers/145filehashes/20trustedV20sources/109dependencies checked.
Original bytes retained; local review experiment_results/v20_failure_review.EXSwDD90.
Only smoke_cache complete (8TRAIN/2dev/2qual_dev), server array verification receipt
passed. A_self first backward failed, empty history and ZERO completed updates.
All other smoke/fullcache/training/evaluation NOT_RUN. Startfree46.90GiB, error
is NOT OOM. No V20 quality/semantic conclusions; do not request training extension.

Exact error adaptive_avg_pool2d_backward_cuda lacks deterministic implementation.
run.py enables strict determinism; model.py Fusion self adaptive pooling triggers
the official PyTorch1.12 guard. Old model contracts used CPU tensors, missing the
CUDA+strict+gradient combination. Implementation validation omission acknowledged.

User explicitly requested repair. Additive docci_text_guided_image_v20_fix1 only;
original V20/old dependencies/manifests/data/cache/weights/results unchanged.
Shared docs exact pre-edit copies deployment_records/v20_fix1_before_20261018_01.
Module-local model.F alias only replaces adaptive pooling with fixed nonoverlap
4x8 block means. Current16/32/64scales same math/gradients; reduction-order float
differences allowed in equivalence test. NO Torch-global patch, detach, warn_only
or determinism disabling; all3arms/count3879760/data/loss/noise/protocol unchanged.

Fix launcher reuses original bounded sequential pipeline via explicit repaired
child entries. Before any cache/model stage, actualGPU1 CUDA gate is mandatory:
pool CPU-reference forward/gradient equivalence and bitwise repeat, all3arm GPU
backward+updates, real LPIPS inputgrad/frozenweights. No skipped gate accepted.
Gate receipt is exclusive and copied into new root with repair snapshot; completed
neural stages must contain exact repair/runtime/strict receipts. Legacy metrics,
bootstrap and numeric thresholds remain unchanged, compatibility provenance added.

--reuse-smoke-cache verifies original source images/arrays, same config and exact
file set, independently byte-copies only12samples into NEWroot/smoke_cache, verifies
again and records old hashes. No old directory writes, hardlinks or symbolic links.
Full8192cache still created for first time AFTER all3real smoke passes. Original
failed train not resumed; new weights start approved parent as intended. No silent
retry, environment install or unrelated job stop. GPU1start20GiB/runtime4/disk64,
max48h budget with original collection reserve. Actual training ETA still unknown.

15newCPU+61oldCPU tests passed, 3newCUDA+6oldTorch explicitly SKIP on local host
without Torch/HF. Python3.8AST,4READMEbashblocks,8archive-member hashes and isolated
restore/source-check+15CPUrerun passed. QA in deployment_records/
v20_fix1_QA_20261018_4xgsi6hh/QA.json; no neural/CUDA/actualsavedweight smoke locally.
Source tar SHA da4285999c36d1206b6c0dcbb578f9f0e843189b115f414b52309b851d7df51d.
Original LPIPS offline assets reused; no weights in repair source tar/GitHub.

Suggested NEW run experiment_results/v20_text_guided_image_fix1_01, reuse original
experiment_results/v20_text_guided_image_01/smoke_cache. One pipeline command only.
If CUDA gate fails before root creation, launchlog may be the only artifact; send
it, do not retry same output/gate identity. Post-stage failures auto collect PRIVATE
light records using Fix1 entry; manual report/collect must also use Fix1, not original.
Public aggregate docs/results/v20_failure_20261018; no raw DOCCI text/images/records
or model weights published. No live server access or remote process started here.

