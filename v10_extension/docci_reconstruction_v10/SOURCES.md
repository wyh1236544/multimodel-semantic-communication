# Primary-source rationale / not a paper reproduction

- SwinJSCC, IEEE TCCN2024, DOI10.1109/TCCN.2024.3424842:
  https://arxiv.org/html/2308.09361v2 ; https://github.com/semcomm/SwinJSCC
  Hierarchical/shifted-window communication encoder/decoder and MSE for the PSNR-oriented model.
  Four-stage public C32 gives complex CBR1/48, versus our512/C8 canvas1/192.
  Published PSNR at a larger rate is NOT a target guaranteed at our lower rate.
  Our own VAE-latent64/96/128 network is inspired, not official architecture/checkpoint replication.
- WITT, ICASSP2023: https://arxiv.org/abs/2211.00937 ; https://github.com/KeYang8/WITT
  Joint pixel encoder/decoder training through stochastic channels, hierarchical Transformer.
  We keep VAE frozen here to first isolate communication codec/objective effects.
  Direct-pixel end-to-end JSCC remains a distinct possible next step if this fixed-VAE path is inadequate.
- Latent Diffusion for Language Generation, NeurIPS2023:
  https://papers.neurips.cc/paper_files/paper/2023/file/b2a2bd5d5051ff6af52e1ef60aefd255-Paper-Conference.pdf
  https://github.com/justinlovelace/latent-diffusion-for-language
  Official autoencoder recipe has frozen BART,32x64 bottleneck, max64 source tokens,
  three-layer compressor/reconstructor, batch256 and50,000 updates.
  The short-text autoencoder scores are BLEU/ROUGE, NOT long-DOCCI/FSO WER.
  This version restores64-token-per-codec difficulty but trains much fewer samples/updates.
  Current low-dimensional64 query is not automatically a bug: official nonlegacy code supports it.
  Keeping full long descriptions by nine public chunks increases rate; no free chunk count.
- DeepSC, IEEE TSP2021, DOI10.1109/TSP.2021.3071210:
  https://arxiv.org/abs/2006.10685
  Trainable Transformer communication layers/CE and short-sentence conditions, not evidence that
  frozen BART32slots can faithfully represent512 tokens at a fixed narrow resource budget.
- DeepJSCC, IEEE TCCN2019: https://arxiv.org/abs/1809.01733
  End-to-end noisy-channel training and graceful degradation motivate our explicitly optional
  matched clean/FSO continuation, not proof that our assumed optical channel matches hardware.
- The Perception-Distortion Tradeoff, CVPR2018:
  https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html
  Plausible sharper diffusion outputs need not improve faithful pixel recovery; no diffusion polishing here.

No external paper code copied into this module. Historical project components remain separately pinned.
No test/qual_test source is loaded for training/validation/model choice; official counts are metadata only.
