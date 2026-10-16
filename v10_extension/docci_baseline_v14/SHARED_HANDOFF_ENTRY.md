# V14 clean single-modality baseline calibration (2026-10-15)

User authorizes implementation and source synchronization, not a remotely started GPU job.
Add docci_baseline_v14; preserve V6—V13, historical checkpoints/results and quality gates.
Offline physical GPU1, existing Python3.8/Torch1.12/Transformers4.30.2; FP32, no installs.

Readonly calibration evaluates approved actual V12 B1 text and approved I2, plus
same-input native BART/VAE privileged references. Clean SNR=null; I2 internal40
sentinel is not measured SNR. No FSO/Common/CLIP/diffusion or digital comparison.

Fresh text legacy64 vs released-code-aligned64 share data/CE/presentation budgets,
32x64 per chunk, all9 frames/9216complex/R1, no RX source length/mask. Paper's
Table13 hidden768 contradicts released default narrow64; optional wide768 is a
capacity hypothesis, NOT official default or a claimed full paper replication.
Unrestricted greedy and deployment beam4/repeat control are separate for all arms.

Image data64 vs expanded start from SAME approved I2 tensors, same updates/LR/
pixel_MSE+.01latent_MSE/4096complex; VAE frozen but differentiable at input.
Pilot2048 and explicit full9647 supported; CPU disk-backed bounded LRU, no truncation.
No automatic promotion or smoke weights reuse; frozen/checkpoint/source audits.

Independent CPU verifier checks matrices, WER/frames, source and actual training
presentations. New private light archives exclude weight/cache arrays; retain actual
text/PNG and honest channel/LPIPS status. Optional explicit offline LPIPS assets,
no automatic downloads. Human facts source-first, blinded, no synthetic fact scores.

Local stdlib/NumPy/integrity tests are separate from mandatory real Torch/HF tiny
contracts and authentic GPU/backbone smoke. Local Torch unavailable; numerical
and recovery results NOT_EXECUTED. No owned school-server PID or training initiated.
Source USTAR/checksum, isolated restore and atomic GitHub source save required.
Initial recommendation: launch.py calibrate NEW --task both, then separate text
pilot, not simultaneous text/image/full/FSO. See README for full commands/limits.
