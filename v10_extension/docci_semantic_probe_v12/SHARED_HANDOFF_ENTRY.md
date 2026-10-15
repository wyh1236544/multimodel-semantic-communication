## 2026-10-15 V12 matched CE/KD and privileged spatial-semantic receiver probe

User authorizes implementing the two proposed diagnostic experiments, not new
server jobs. Add docci_semantic_probe_v12 only; preserve V6-V11 sources, weights,
data, failed/results and all historical quality gates. Default physical GPU1,
existing offline Python3.8/Torch1.12/Transformers4.30.2/Diffusers0.20.2 only.

Text: approved V11 final20480 weights split into NEW-optimizer CE and CE+full-vocab
TRAIN-only frozen-BART KL,4096 updates each, matched init/order/LR/batch2/TRAIN2048/
qual_dev100/9216complex/R1/unrestricted greedy generation. Teacher/source memory,
gold prefixes/masks/length never actual RX inputs. No FSO/Common/CLIP in text task.
Keep old WER/margin/EOS quality gates; prefix diagnosis uses published FP32
roundoff-bound verifier, not a silently weakened quality threshold.

Image: approved I2 final512 checkpoint and VAE frozen, same399843-parameter
three-scale receiver residual modules. A0 uses image-derived self context only;
A1 adds privileged original-description Long-CLIP vector. BOTH use clean/mask25
alternating TRAIN2048,2048 updates/accum2, identical fixed4096complex/R1 payload.
Shared stored validation observations for original I2, trainedA0 and A1 matched/
shuffled/zero semantics. Full Long-CLIP246-token windows, weighted pooling, no
truncation. Text side info is NOT transmitted/charged: not end-to-end or equal-
total-budget joint communication. Mask25 is synthetic feature erasure, NOT FSO.
PSNR predeclared primary; SSIM and all cases kept; LPIPS NOT_MEASURED, no downloads.
No HiFiC/GAN/diffusion or bidirectional-Common success claim.

CPU contract/report tests and actual historical stored-frame regression executed
locally; Torch/HF unavailable,6 model tests SKIPPED, real saved-backbone/512/GPU
smoke still mandatory. launch requires actual tests -> discarded fresh smoke ->
independent CPU acceptance -> NEW main process from approved parent, finite cap,
no auto retry/promotion/other-process stops. Final CPU collection separate, original
run read-only, no legacy180s report cutoff. README has exact upload/install/GPU1/
run/progress/collect commands, forecast1-3h NOT measured, cap6h not ETA. New source
tar/SHA and atomic GitHub synchronization required. No server task PID to report.
