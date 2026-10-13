# Verification scope

Local environment: bundled Python3.12 + NumPy/Pillow, no Torch/Transformers/Diffusers/CUDA/DOCCI/officialLongCLIP weights. No environment changes or server training performed.

Executed locally before packaging:

- All new Python syntax parsed for Python3.8.
- Eight synthetic core contracts: official DOCCI schema/split separation, noCOCOfallback, no silent BART truncation, ordered246 Long-CLIP chunks, native full-frame letterbox preservation, write/symlink guards, exact256/512 C/I/T budget, proxy/fixed-noise reference and an actual6400-payload V3 optical waveform through receiver with precise time/energy accounting.
- Five independent report tests: recomputed full-source text metrics, strict full matrix, quality-stop vs smoke/full completion, forbidden clean-source field in RX frames, path/symlink rejection. Synthetic data is explicitly not a real model recovery result.
- Real deterministic source-archive restore, additive install, idempotent reinstall, exact prior HANDOFF backup, source conflict/no mutation, prerequisite drift rejection, symlink destination rejection, and execution of restored core/report tests. Build-verification JSON gives actual outcomes and hashes.

Not executed locally: `test_model.py` tiny Torch fixtures, authentic model load/gradient/checkpoint/GPU smoke, real DOCCI length audit,512 peak GPU memory, optical task recovery, joint gain, runtime benchmarking. Missing Torch is reported as NOT_EXECUTED, not passed. Server launcher requires Torch/Transformers tests to actually run and then authentic256/512 smoke. No dependency upgrade is requested.

Evidence requirements:

- A root `SMOKE_COMPLETE.json` establishes engineering execution at that size only, irrespective of8-step quality. `QUALITY_STOP.json` is deliberately not `COMPLETE.json`.
- Main COMPLETE only follows positive independent verification of data/source/frozen identities, private-text gate, group nonzero gradients/weight changes, checkpoint roundtrip, full physical eval matrix, same-frame interventions, paired noise seeds and charged public resources. It does not imply joint improvement or publication success.
- Results show PSNR/SSIM on unpadded512 full view and full-original WER/CER/exact/ BLEU2; black/empty receiver failures count. PNG-based checks accommodate explicitly bounded8-bit quantization and do not silently replace float metrics.
- Do not compare absolute PSNR against historical128/COCO targets as an architecture gain. Single training seed/qual_dev confidence intervals cannot establish generalization over retraining or independent test.
