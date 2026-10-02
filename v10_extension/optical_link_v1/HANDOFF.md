# Optical Link V1 handoff

2026-10-02. User authorized implementation after reviewing the physical-link
proposal; existing user rules require tar.gz, server extraction commands and
GitHub source preservation. GPU1; no environment upgrades; preserve history.

Added only optical_link_v1/. Frozen A uses the exact original V10 base and A
delta (validated SHA/split/parameter names). No neural training or external GPU
process modification. New SI optical noise/power mode has no nominal-SNR knob.
README.md is the canonical component/units/assumptions/commands contract.

Local NumPy/bit-protocol tests, random-byte diagnostics and release restore are
distinct from actual Torch/GPU/A-weight/COCO image/text validation. This machine
has no usable PyTorch; do not install/upgrade the user's server environment.
Actual performed checks are in VERIFICATION.md and release verification JSON.

Next user-started server steps: install, check, link diagnostic, 2-image smoke;
inspect power/SNR/BER, all recovered text and images including failures; then
8-image quick if runtime and outputs are correct. No test split or new training.
Return collect.py small records or failure log/run_identity. Parameter values
are assumptions until hardware calibration, not OptiSystem numerical equivalence.
