# V10 optical-link rate and constellation pilot

Status: isolated **validation-only exploratory code**. No new experimental result is
claimed until the user's server completes all runs and `rate_summary.json` is
independently checked. This directory adds no model weights or data and does not
change V10-A training/evaluation source files or existing outputs.

## Exact questions and boundaries

1. `imdd_continuous`: original V10 Gamma-Gamma/pointing IM/DD differential
   intensity pair, with 9,408 forward optical-intensity slots for a full frame.
2. `coherent_continuous`: same FSO *irradiance* draw, square-rooted into complex
   field amplitude with random block-constant phase; 32 complex pilot symbols
   estimate the gain/phase. A complex-field semantic payload uses 2,304 symbols;
   with pilots/header the forward total is 2,400 complex symbols.
3. `coherent_qpsk` and `coherent_16qam`: the **same** coherent channel, checkpoint,
   image IDs and random seeds as `coherent_continuous`, but transmitter maps each
   continuous complex semantic value to a finite constellation and receiver makes
   a hard symbol decision after pilot-aided equalization. These are uncoded
   constellation-constrained semantic pilots. They are **not** a conventional
   compressed-bitstream + FEC communications baseline. No retraining is done;
   quantization mismatch can be large and must not be described as a method limit.

The reference SNR for original IM/DD is average optical intensity squared over
thermal-noise variance. Coherent SNR is complex symbol energy over complex
noise variance. **Equal nominal dB does not imply equal photons, optical power,
or receiver sensitivity.** Report the within-coherent constellation contrast;
show the IM/DD curve as an architecture reference, not a matched-power winner.

The rate sweep assumes fixed received power and white thermal-noise PSD; when
symbol rate is multiplied by `r`, nominal SNR falls by `10 log10(r)` dB. This is
a declared normalized engineering assumption, not measured hardware. Forward
serialization is `forward_uses / forward_symbol_rate`; the separate reverse RF
feedback is `64 / feedback_symbol_rate`. The displayed pairs/second is an
**ideal sequential link-only upper bound**, excluding VAE/CLIP computation,
packet guard, synchronization, DAC/ADC limits, pulse shaping, shot noise,
laser/LO phase drift, optical coupling, MZM transfer curve, safety limits and
retransmission. Analog transmission has no defined application bit/s without a
quantizer; finite-constellation gross label bits are not application goodput.

This small pilot uses the **existing V10 checkpoint without retraining**. It
cannot establish a publication-grade modulation comparison. Follow-up requires
matched-power optical calibration, phase-noise/receiver model, separately
trained constellation-constrained models (or a real codec+FEC baseline), more
images/repeats/seeds, and frozen test evaluation after validation choices.

## Install without overwriting the existing project

From the Mac, upload the delivered `v10_link_compare_v1.tar.gz` to the existing
server project. On the server, first check `v10_extension/link_compare` does not
exist; then inspect and extract the archive from the project root. All archive
members are under that one new directory. Do not use `tar --overwrite`.

## Server preflight and pilot

Run from `/data1/zw/WYH/multimodel`. Use the server's existing Python and GPU 1
only if its free memory is adequate and the process belongs to your allocation.

```bash
nvidia-smi -i 1 --query-gpu=index,memory.free --format=csv
test ! -e experiment_results/v10_link_compare_pilot_v1
/home/zw/anaconda3/envs/torch/bin/python -B -m unittest v10_extension/link_compare/test_link_compare.py
CUDA_VISIBLE_DEVICES=1 /home/zw/anaconda3/envs/torch/bin/python -B v10_extension/link_compare/run.py \
  --checkpoint experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth \
  --split-manifest v10_extension/manifests/ln_v10_splits.json \
  --output-dir experiment_results/v10_link_compare_pilot_v1 \
  --data-root COCO2017 --vae-path pretrained/sd-vae-ft-mse \
  --text-model-path pretrained/clip-vit-base-patch32 \
  --profiles fso_moderate --snr-list 0 10 20 \
  --symbol-rates 1000000 10000000 --reference-symbol-rate 1000000 \
  --feedback-symbol-rate 1000000 --samples 8 --repeats 1 --save-images 4 \
  --max-hours 0.5 --max-total-hours 2
```

`--dry-run` appended to the final command prints the plan without importing
PyTorch or writing. The command refuses an existing output root. Any failed
partial run is preserved; retry with a new output directory name.

Only after a real pilot completes, read `rate_summary.csv`,
`rate_summary.json`, each child run's `summary.json` and `samples.jsonl`, and
inspect both received images **and** received captions. Report all cases,
including degraded examples; don't select only successful reconstructions.

## References and design choices

- [DeepJSCC-Q, finite-alphabet JSCC](https://arxiv.org/abs/2206.08100):
  motivates a within-physical-channel constellation contrast, not a claim that
  our zero-shot implementation reproduces its trained method.
- [Coherent versus IM/DD FSO comparison](https://air.unipr.it/retrieve/a533455a-838f-4141-a6b2-76925dc5b1d6/An_Information-Theoretic_Comparison_between_Coherent_and_IM_DD_Transmissions_for_Free_Space_Optical_Communications.pdf):
  motivates keeping detection and optical power assumptions explicit.
