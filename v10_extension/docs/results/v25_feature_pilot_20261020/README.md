# V25 frozen BLIP-2 feature-to-BART pilot: aggregate evidence

2026-10-20. Engineering finished; the prespecified content-specific WER screen did
NOT pass. Raw DOCCI descriptions/images, private pages, records and weights are
not published here. No final-test or general multimodal-impossibility claim is made.

The PRIVATE pilot archive SHA256 is
`d9478e118dbded2bbb602245ad6c567b302efdc38e3af2a4b7249172c1fb3774`.
All 237 listed hashes/238 safe archive files were checked. Four source snapshots
matched pinned V25 source. Server 15 CPU/6 older HF/8 new real HF tests passed;
frozen tensor checks, nonzero learning groups and checkpoint roundtrips passed.
Local review could not rehash the weights/NPZ kept on the server. The exact local
legacy verifier stopped on eight BLEU2 differences of at most 1.39e-17; a separate
documented arithmetic audit retained exact WER/error counts and all other checks.
Neither raw results, old source nor the prespecified success threshold was changed.

## Fixed setup and results

512 paired TRAIN items, 32 reused TRAIN-source development items, one trained seed,
4 epochs/512 optimizer updates per arm, 1,516,290 trainable parameters in each
matched-size adapter. V20 image/V18 text codecs, VAE, BART and BLIP-2 remain frozen.
Cached channel observations repeat across epochs; this is not fresh-noise end-to-end
JSCC training. No original images or captions are visual receiver inputs.

Mean normalized WER per description (lower is better; not semantic accuracy):

| Image / text condition | Frozen text | Same-size self | Correct image | Wrong image |
|---|---:|---:|---:|---:|
| clean / clean | 31.96% | 32.70% | 32.49% | 32.74% |
| 0dB / 0dB, PRIMARY | 70.94% | 70.40% | 71.41% | 71.02% |
| clean / 0dB | 70.94% | 70.40% | 71.23% | 70.86% |
| clean / -5dB, out of training range | 88.46% | 87.22% | 87.78% | 87.38% |
| -5dB / -5dB, out of training range | 88.46% | 87.22% | 87.33% | 87.39% |

AWGN only, not FSO or a 64GSa/s waveform experiment. Clean has no added noise,
but still has lossy codec reconstruction; it is not original data or a finite20dB.
Joint frame is 13,376 complex symbols, R1, including both32-pilot groups; image-only
caption input is4,128, text-only9,248. Adapter-size matching does not equalize full
system compute or give visual information for free.

Primary signed gain is control WER minus correct-image WER, in percentage points:

| Control | Mean gain | Paired95% percentile interval |
|---|---:|---:|
| Same-size self | -1.003 | [-2.144, +0.006] |
| Wrong image | -0.383 | [-1.508, +0.555] |
| Frozen text | -0.466 | [-1.870, +0.889] |

10,000 example-paired bootstrap draws, fixed PCG64seed250042. One trained seed,
reused exploratory development set, fixed cached noise, pointwise intervals without
multiplicity correction; not final publication significance. Primary intervals
include zero, so they do not establish harm either. The prespecified positive
gain thresholds were not met. Repetition was15.15% for correct visual vs14.03%
for self, also failing its safeguard. Empty-slot activation was zero in both.

Correct vs wrong image changed generated tokens in23/32 primary examples, but WER
improved in7, tied in15, worsened in10. The branch affects decoding, yet useful
content-specific correction is unproved. At-5dB the visual gain over frozen text
was about1.14pp, while self gained about1.24pp; the correct/wrong gap was only0.07pp.
That is not evidence that correct image semantics caused the improvement.

Factual retention, contradictions and hallucinations were not independently scored.
The original facts=PENDING, semantic_success=false and failed numerical screen
are retained; word error alone cannot rule out every local semantic improvement.

## Runtime and next action

Cache1096.40s; self training/audit137.57s; visual133.13s; evaluation616.71s;
PIPELINE2453.62s excludes later final verification/collection. Cache GPU peak
allocated8.91067GiB/reserved10.10547GiB. These are measured V25 values, not V26
memory guarantees. Full wall time is not equivalent to optimizer training time.

V26 is an additive read-only diagnosis: existing checkpoints/caches, training-vs-dev
content TF CE, effective residuals/receiver-only gates, limited TRAIN greedy output
and explicit reuse of existing DEV output. It does not retrain or change this result.
New real V26 GPU/HF measurements are NOT_EXECUTED at code delivery.
