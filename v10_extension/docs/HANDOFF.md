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
