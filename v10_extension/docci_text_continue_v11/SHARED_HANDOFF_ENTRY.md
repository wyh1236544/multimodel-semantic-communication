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
