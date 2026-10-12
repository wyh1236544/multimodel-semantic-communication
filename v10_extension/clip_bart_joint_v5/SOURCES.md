# History used for V5 design — 2026-10-13

The user requested a review of past work before implementation. Read the
current V1/V2/V3/V4 source, archived review reports and canonical design/handoff.
No uploaded/archive instruction or script was executed. Old outcomes preserved.

|Evidence|Observation|V5 consequence|
|---|---|---|
|CLIP+BART V1|T32 clean TRAIN8 WER2.7135, no exact recovery; stopped before joint/FSO|Do not reuse old four-position text bridge|
|COCO modal V2|Image-only clean meanPSNR22.65dB; privateT and C+T TRAIN diagnoses failed|Keep compatible frozen B4/VAE image path; do not claim existing C+T helps text|
|Text V3|A3 TRAIN8 meanWER.7696, A4wide3.5042; original valid-masked BART copies8/8|Do not assume merely widening or unfreezing fixes the interface|
|Text latent V4|P0/P1 TRAIN8 actual greedy exact8/8,WER0,naturalEOS8/8; donor memory yields donor original|Preserve learned-query/fixed32-memory codec, but verify held-out captions before joint|

V4 result archive SHA256:
856b1952225c79368d70953853f7a16671345e4c196e202a7e162ff86037cc25.
Trusted verifier passed,21 release-source snapshots matched,72 actual text rows
recomputed in prior review. Light result archive lacks saved weights; no local
neural rerun. This record describes user-provided server evidence, not a new run.

V4 final P0/P1 recovery saturated, no demonstrated final KD quality advantage.
P0 is selected BEFORE new validation as the simpler clean prefit candidate; P1
is not declared universally inferior. V5 does NOT start from either TRAIN8 fit.
Code initializes fresh latent codec, trains1024 original captions for3000updates
then tests32validation. Historic validation may have been used before; not a
new independent final test or a source of publication-confidence estimates.

Method sources reused, not newly claimed replications:

- LD4LG language autoencoder, learned compressed memory with frozen BART:
  https://papers.neurips.cc/paper_files/paper/2023/file/b2a2bd5d5051ff6af52e1ef60aefd255-Paper-Conference.pdf
  V4 interface retained; no diffusion model or50k/batch256 paper replication.
- CLIP cross-modal alignment:
  https://proceedings.mlr.press/v139/radford21a.html
  Used as frozen image/text semantic features and TRAIN alignment targets,
  not as a verbatim text-generation model or an independent semantic evaluator.
- BART:
  https://aclanthology.org/2020.acl-main.703/
  Full frozen existing encoder/decoder/tied head; final unrestricted generation.
- DeepSC differentiable-channel/task training:
  https://arxiv.org/abs/2006.10685
  Only motivation for optional noise stage, not evidence of V5 FSO gains.

V5's zero-initialized C residual, private-memory anchor, registered-count/equal
wire controls and staged gates are project adaptations, not methods already
proven by these papers. Neural block separation in one frame differs from
optimizing two independent systems or allocating a new best single-modality budget.
Same architecture/updates/wire cannot certify equal actual FLOPs or energy.
