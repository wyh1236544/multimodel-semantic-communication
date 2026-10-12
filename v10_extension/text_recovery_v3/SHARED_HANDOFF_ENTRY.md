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
