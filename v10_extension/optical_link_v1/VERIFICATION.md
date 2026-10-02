# Verification scope

2026-10-02. Local interpreter: bundled Python3.12.14, NumPy2.3.5. Production
target remains existing server Python3.8 / Torch1.12 / NumPy>=1.17, no upgrades.

Executed local numerical suite: 32 tests covering configuration, unit conversion,
Gamma-Gamma moments, Rytov mapping, exact circular Gaussian aperture collection,
sqrt irradiance-to-field scaling, EDFA gain/ASE/total-power limit, sampled PSD and
ENBW, hybrid energy conservation, passive IQ modulator, per-PD thermal/shot noise,
null undefined/noiseless SNR, received-pilot CFO/SNR,
QPSK AWGN BER reference, full C/I/T quantized CRC roundtrip, full-object erasure,
linear SNR aggregation, coherence cap, all-failure-inclusive report, SVG panels,
duplicate result cells and packet-evidence reconciliation.
Python3.8 grammar parse covers every new .py file, not runtime execution in3.8.

The first test attempt found only a float64-vs-float32 reference assertion issue;
the reference is now compared after the same explicit float32 output conversion.
No BER/CRC acceptance or physics threshold was relaxed.

Random-byte CPU link diagnostics exercise all three configured physical cases,
including accepted packets and deep-fade CRC failures; not neural image results.
Source archive builder independently extracts and hashes every source file,
checks fresh additive installation, identical reinstall, conflict preservation
and missing prerequisites. Build verification JSON records the actual artifact.

NOT executed locally: test_model.py (requires Torch); actual V10+A base/delta
loading; GPU1/COCO/CLIP/VAE inference; server Python3.8/Torch1.12 runtime;
hardware calibration; actual OptiSystem comparison; neural improvement/training.
Server launch.py check must run both core and Torch adapter suites before smoke.
Failures and partial records are preserved, not relabeled as completed results.
