# Token-guidance quick V1 handoff

2026-10-01. User explicitly authorized implementation of a quick joint-token
direction-validation experiment, GPU1, keeping all old adaptive/business paths.
Canonical assumptions, commands and inference boundary: README.md.

Added only token_guidance/. Four independently short-fine-tuned factorial arms,
original-weight control, exact waveform/FEC forward with disclosed STE backward,
no raw UTF8 receiver prompt, clean quantized reference, sender-only and
received-memory-only mismatching and refiner bypass. All fixed resources and
paired field/noise IDs audited; no test use, no best-checkpoint selection.
Four delta weights require exact preserved V10 mixed seed42 base SHA.

Local environment provides NumPy but no usable PyTorch. NumPy transport/report
tests and Python3.8 syntax checks have passed. Package restore/conflict checks
are executed by the release builder. See VERIFICATION.md for exact coverage.
Torch CPU hook tests and actual
COCO/VAE/CLIP checkpoint training/gradient/GPU effectiveness require the existing
server Python3.8/torch1.12 environment. Do not treat CPU/syntax success as model
improvement. No remote training was launched or external GPU process modified.

Next: install/check, four-arm8-image smoke, inspect timing/gradient/CRC completion,
then user-started512-image2epoch quick. Return collect.py records for direction
decision; do not automatically expand to formal many-seed training.
