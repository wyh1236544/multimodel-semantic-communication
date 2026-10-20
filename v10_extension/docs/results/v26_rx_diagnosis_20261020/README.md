# V26 actual read-only receive diagnosis

Verified 2026-10-20 from the PRIVATE uploaded light archive. Archive SHA256:
`ff273c2769859721acf2b3c470c3f0b5604fca195acc57504ae7da8eb10628c8`.
84 regular files, 83 collection hashes, 3 executed-source snapshots and the pinned
workspace pipeline verifier passed. Six real tiny-BART and eleven CPU contracts
were recorded as passed on server; each of smoke/diagnose reproduced eight actual
saved-parent generation anchors. No parameters changed. Heavy weights were not
downloaded or independently replayed locally.

## Main finding

The visual branch operates but has not shown stable correctly paired content
benefit. Under the predefined image/text AWGN 0 dB condition, the DEV content-gate
mean is 0.995793; the two layers' mean effective content-position residual ratios
are 5.70% and 8.83%. These are teacher-forced positional measurements, not the
residual of the entire free-generation path or proof of semantic correctness.

| Fixed cohort | Correct-vs-wrong content CE gain | Pointwise paired 95% CI |
|---|---:|---:|
| 16 TRAIN examples | +0.000541 | [−0.000722, +0.001896] |
| 32 reused DEV examples | −0.000308 | [−0.001233, +0.000579] |

Gain means wrong CE minus correctly paired CE, averaged equally by description.
Neither interval establishes a stable paired benefit. This is not an equivalence
test and cannot prove zero effect. Gold prefixes are used only in TF diagnostics,
not in blind reception. Lack of benefit even with those prefixes makes generation
repetition an insufficient unique explanation; representation, objectives, data
and optimization remain candidate causes, not a proven single cause.

DEV generation is reused from V25, **not a new independent replication**:

| Method | Mean normalized WER |
|---|---:|
| Frozen text receiver | 70.94% |
| Equal-parameter self adapter | 70.40% |
| Correct visual assistance | 71.41% |
| Mismatched visual assistance | 71.02% |

Main correctly paired WER gains over self and mismatched visual inputs remain
−1.003 pp and −0.383 pp, with both paired intervals crossing zero. The old V25
screen remains failed. Factual recovery remains unmeasured/PENDING, not assumed
to be zero based on WER.

## Execution and limits

Diagnose: 16 TRAIN, 32 reused DEV, five historical conditions, seven controls,
1,680 TF calls, 560 newly measured TRAIN greedy outputs, 1,120 reused DEV outputs.
No training, fresh transmission/noise, BLIP/VAE/source codec load, optimizer or new
checkpoint. Five conditions remain AWGN; −5 dB is out-of-training stress, not FSO.
The inherited frame allocation is 13,376 complex symbols including pilots, R1;
individual minimum input resources differ by method.

Diagnose took 519.93 s; Torch peaks were 0.831 GiB allocated / 0.947 GiB reserved.
718.44 s to PIPELINE COMPLETE excludes subsequent final verification and collection.
The post-collection final launcher marker is not included in the light package.
Local parent reading recorded eight BLEU-only ~1e-17 roundoff notices, exact WER
and counts, with no numerical summary changes or modification of old code/results.

Single fixed weights, n16/n32, reused development cohort and pointwise uncorrected
10,000-draw paired bootstrap imply exploratory evidence, not final significance.

## Next implemented step

[V27](../../../docci_fact_bridge_v27/README.md) prepares an offline source-fact and
candidate review, not another neural training run. Source facts freeze before the
new candidate-rating phase; prior exposure and blinding limits are recorded.
No automatic factual approval, image-ground-truth claim or training launch.
Raw source/recovered descriptions, HTML, private records, labels and weights are
not included in this public repository.
