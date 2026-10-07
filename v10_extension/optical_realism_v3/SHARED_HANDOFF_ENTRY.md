## 2026-10-07 optical realism V3 receiver repair

Additive optical_realism_v3 preserves V2, historical weights/results and split.
Public even-pilot coherent bounded CFO search (default +/-250MHz), joint
buffered clock fit, smoothed complex pilot phase and odd-pilot validation replace
noisy unwrap/random-walk acquisition. The synchronization score has a new
held-out coherent definition, not merely a lower V2 threshold. No hidden
CFO/clock/fading/source-feature access.

Analog reliability retains an unscaled held-out block-error floor and split-half
repetition inconsistency as well as white variance/R. It is an uncalibrated
conservative heuristic, not certified delivery. Whole-coded-block 16QAM repeat/
LLR combining replaces adjacent pairs: all 16 points are reachable, all repeated
bits/pilots/samples charged. JSON/Markdown and effective preset identity repaired;
pre/post-gate constellations, separate waveform vs RX-selected projection SNR,
probe/payload lock and accepted-but-MSE>1 diagnostics retained.

Local 140 NumPy/protocol/legacy regression tests, 15 B2B functional cases and
24 same-ADC synthetic V2/V3 comparisons executed (exit0). Combined moderate
synthetic mean feature MSE 1.269 -> .066; stress .753 -> .276; good .00354 -> .00367
(not uniformly improved). No V3 accepted MSE>1 in this small synthetic set.
No Torch, real A/COCO/GPU1, V3 image/text gain or hardware calibration executed
locally. Original A text ceiling unchanged.

Source-only tar/checksum, restore/install/conflict checks, additive installer and
atomic GitHub save required; installation backs up HANDOFF before append.
Next user-started step: optical_realism_v3/launch.py pipeline in a NEW project
output, GPU1, --max-hours 1: tests/Torch hook -> B2B -> synthetic replay ->
2-image smoke -> two 8-image quick contrasts -> verified record bundle.
No training, no seed sweep, no automatic full validation/test. Return
v10_optical_realism_v3_records.tar.gz, then decide on middle power points.
No server jobs launched by this task. See optical_realism_v3/README.md and
VERIFICATION.md for commands and limits.

