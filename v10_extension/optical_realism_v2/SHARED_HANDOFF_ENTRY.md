## 2026-10-05 optical realism V2

Additive `v10_extension/optical_realism_v2`: frozen A, no training, GPU1,
validation-only. SI optics plus clock-before-ADC/ENOB/IQ imperfections, public
interspersed-pilot timing/CFO/phase receiver for arbitrary complex payloads,
explicit rejection, fixed public-calibration laser working point, charged pilots
and repetitions, QPSK-convolutional vs 16QAM-convolutional-repeat2 comparisons.
Same device/peak-envelope/slot caps do NOT mean equal actual average power/energy.
Default pipeline is check -> 15 B2B diagnostics -> 2-image smoke -> TWO 8-image
quick runs -> verified image/text/constellation record bundle. Not formal test.

Local NumPy contracts/B2B/regression checks executed; actual Torch hooks,
base+A delta loading, GPU/COCO performance, hardware calibration NOT executed
locally. No server tasks launched. Historical source/weights/results preserved.
Source tar + checksum + additive installer + GitHub atomic backup required.
See optical_realism_v2/README.md and VERIFICATION.md for commands and limits.
First return: v10_optical_realism_v2_records.tar.gz from a NEW pipeline output.
