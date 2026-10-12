# Method provenance and explicit deviations

- Lovelace et al., Latent Diffusion for Language Generation, NeurIPS2023:
  https://papers.neurips.cc/paper_files/paper/2023/file/b2a2bd5d5051ff6af52e1ef60aefd255-Paper-Conference.pdf
  Section3.1 language autoencoder, Table1, Appendix Tables10/13. Borrow learned
  query compression, fixed semantic memory, frozen pretrained encoder/decoder,
  CE objective. No diffusion model implemented. Reported BLEU97.6/98.8 is NOT
  exact recovery rate or FSO performance. Paper50k updates/batch256, not our screen.
- Author implementation (read-only inspected 2026-10-12):
  https://github.com/justinlovelace/latent-diffusion-for-language/blob/main/latent_models/perceiver_ae.py
  https://github.com/justinlovelace/latent-diffusion-for-language/blob/main/latent_models/latent_finetuning.py
  MIT author repository; this module is an independently written adaptation,
  not a vendored copy. Differences: max128 COCO short captions, different budgets,
  compact wire transport, paired CE/KD screen, gradient/frozen/real reload gates,
  unconstrained greedy primary and constrained beam secondary, FSO continuations.
- Kim & Rush, Sequence-Level Knowledge Distillation, EMNLP2016:
  https://aclanthology.org/D16-1139/
  Word-level teacher distributions motivate our optional training-only KD.
  Original paper does not prove BART compression/FSO gains. Gold text retained;
  no teacher pseudo-text replaces original labels, no teacher features at RX.
- Xie et al., Deep Learning Enabled Semantic Communication Systems, IEEE TSP2021:
  https://arxiv.org/abs/2006.10685
  Motivates differentiable channel/task joint optimization, not a reproduced FSO
  system. Current full waveform is eval-only; symbol proxy is an approximation.
- Exact server-compatible BART implementation:
  https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/models/bart/modeling_bart.py
  Official shift_tokens_right, encoder_outputs and autoregressive generate/cache.
  No library upgrade or new package install. Full offline existing BART is frozen.

Learning a language codec, measuring FSO robustness and integrating multimodal
C/I/T are three separate claims. This package tests the first, and optionally
screens the second; it cannot certify the third, physical hardware or novelty.
