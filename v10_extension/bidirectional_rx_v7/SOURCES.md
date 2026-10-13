# Architecture rationale and boundaries

Implementation is a small independently initialized residual receiver, not a reproduction of pretrained ALBEF/BLIP-2 and not imported paper weights.

- ALBEF, NeurIPS2021: token alignment and task-conditioned fusion motivate explicit cross-modal receiver routes. Not channel training: https://proceedings.neurips.cc/paper_files/paper/2021/hash/505259756244493872b7709a8a01b536-Abstract.html
- BLIP-2, ICML2023: frozen backbones + trained connector motivate keeping existing backbones fixed. Our small adapter is NOT a pretrained Q-Former: https://proceedings.mlr.press/v202/li23q.html
- DeepJSCC-WZ: receiver-side information motivates paired same-damage assistance tests. Its side image and AWGN assumptions differ from our noisy text and FSO: https://arxiv.org/html/2310.04311v2
- VLF-MSC preprint: shared image-language visual features motivate future Common improvements, not exact original-text reconstruction claims: https://arxiv.org/html/2511.10074v1

V6 observations motivating this isolated experiment: private text9000 mean WER about0.137, joint V6 J1 did not beat J0 overall. Therefore preserve the private codec and first test RECEIVER assistance. No claim that all multimodal information is already encoded in Common. No new diffusion or fully unfrozen backbone in this first test.
