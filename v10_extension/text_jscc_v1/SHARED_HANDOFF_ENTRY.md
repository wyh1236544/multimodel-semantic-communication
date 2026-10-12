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
