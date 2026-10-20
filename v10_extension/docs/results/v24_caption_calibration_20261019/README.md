# V24 Fix2 actual caption calibration: aggregate evidence only

Date: 2026-10-19. Engineering checks passed; multimodal recovery is NOT established.
Source archive and prior failed runs are unchanged. Raw DOCCI examples, descriptions,
model outputs, private HTML, records and pretrained weights are not published here.

## Verified provenance and engineering

The uploaded PRIVATE records archive SHA-256 is
`5e702c14b31df25c08053c7e223fec3ea9bf158aebfe5871edcc20eec2b48062`.
All 55 collection-listed files matched their checksums; all 56 archive members were
safe ordinary files. The trusted, already-pinned V24 Fix2 verifier passed independently;
code supplied inside the records was not executed.

Server logs contain six passing real Hugging Face contracts in the original
Torch 1.12.0 / Transformers 4.30.2 environment. This includes actual tiny-checkpoint
save/reload and FP16 metadata JSON serialization. Frozen full-model tensor hashes
matched before and after inference. Training updates: zero.

Twenty condition rows reuse nine actual generations on two previously explored
development images. They are NOT twenty independent examples. All nine generations
terminated naturally; none was empty or capacity-truncated. This says nothing by
itself about factual correctness.

GPU1 peak allocated memory was 7.04474 GiB, reserved 7.31445 GiB. The nine generation
calls took 32.1213 seconds; that is not full model loading, hashing or end-to-end
wall time. These values are not the memory or time cost of the new V25 combined cache.

The actual metadata conversion involved `config.torch_dtype`. Its float32 metadata
value does not mean the whole model ran in float32: the run requested float16, and
actual per-stage types were recorded by forward probes. Only copied metadata was
converted; model configuration objects, parameters and generation settings were not changed.

## What can and cannot be concluded

The two original images and their clean received versions yielded relevant main
subjects, but received-image attributes could still be wrong. At 0 dB there were
subject and attribute errors; at -5 dB both examples lost the correct main subject.
A constant gray image also elicited unsupported content. Wrong-image conditions
reused the corresponding actual image's generation and are not additional trials.

This supports basic operation and some visual conditioning in clean examples. It
does not demonstrate faithful long-description recovery, robustness across a
population, text correction, joint recovery, or an advantage over digital transmission.
No confidence interval or significance claim is made for two reused examples.
WER, BERTScore and factual F1 were not newly measured in this caption calibration.
The original `facts=PENDING` and `semantic_success=false` remain unchanged.

Conditions are unit-gain AWGN, not FSO. Clean means no added noise, not a finite
20 dB channel and not lossless image reconstruction. The -5 dB condition is outside
the frozen codec's training range. Original images are privileged diagnostics only;
the captioner was not given received text. Caption-only uses the 4,128-complex-symbol
image frame, not the joint image/text frame of 13,376 symbols.

The next separately authorized V25 feature-adapter smoke is a new experiment.
It does not retroactively turn these caption-only results into evidence of mutual help.
