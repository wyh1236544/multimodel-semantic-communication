# Source and initialization provenance

No new literature efficacy claim or external model is introduced here. This revision intentionally restores independently decodable private-branch baselines before evaluating joint semantic gain.

- BART codec architecture and P0 clean CE/warmup recipe: existing `bart_latent_codec_v4`, whose project references document Perceiver-style learned semantic slots. This is not a new paper reproduction or native BART representation sent without compression.
- Text weights: own `v10_clip_bart_joint_v6_03/v10_screen/text_prefit/clean/last.pth`, SHA256 `fa739ff95ae3c1e54190524f2ef169235d9cc16eefefceabf670a398d36e3ff7`. Historical fixed qual_dev COCO32 matchedWER0.13717247, after global9000 updates. These historical data are not DOCCI comparisons and are not part of the V9 evaluation cohort. Weights-only transfer, no optimizer replay or claimed exact resume.
- Image weights: own `v10_docci_longclip_joint_v8_root_01/v10_diagnostic/prefit/image/last.pth`, SHA256 `af53e27c71c0fafcb66454a4ed889e5fd487d6ac6314c36c1b960f5fda8d7196`. Only whitelist `image_encoder.` and `image_decoder.` tensors copied. V8 updated64image optimizer steps; its random/failed text and Common state do not initialize V9.
- Backbone files are exactly those used with the two pinned historical checkpoints. Frozen offline BART-base and sd-vae-ft-mse only; Long-CLIP preserved but not loaded. Image transmission baseline is learned V8 spatial encoder/decoder, not a direct VAE oracle and not diffusion generation.
- DOCCI original annotation view and native letterbox handling reuse V8 immutable source utilities. Complete official TRAIN/qual_dev are audited; test/qual_test only counted for split integrity and never given as model inputs.
- Training FSO is the existing V4 differentiable pilot-estimated symbol proxy. Actual optional optical evaluation reuses existing V3 waveform/DSP and fluctuatingV2 correlated paths with new private shorter frames. Approximate-training versus actual-waveform evaluation is explicit; no hardware-calibration claim.

See dependency checksums in MANIFEST and source snapshots in each run for precise reproducibility. No pretrained/source-data downloads are authorized by this revision.
