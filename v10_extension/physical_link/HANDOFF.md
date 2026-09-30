# Physical-link V2 handoff — 2026-09-30

Status: additive implementation, NumPy numerical tests passed locally; torch adapter and real COCO/GPU pending on school server. No server processes launched or modified. No old files/results/checkpoints/splits replaced.

Reference: project conversation “解读实验室代码”, user excludes KK. Ports Gray bit mapping, RRC 0.01, 14GBaud/64GSa/s, 8-bit DAC. Adds source serialization/CRC32 and optional terminated K7 rate-1/2 convolutional FEC; not LDPC. Uses fractional-phase sampled RRC span128 (lab span16), pilot scalar coherent I/Q receiver, not full lab blind/Volterra DSP.

New files: digital.py, waveform.py, channel.py, run.py, report.py, test_core.py, test_adapter.py, install.py, run_pilot.sh, deploy_and_run.sh, README.md, __init__.py and this handoff. digital/core NumPy only; channel adapter imports the unchanged V1 coherent module and existing V10 code. Evaluation-only (detached CPU waveform computation), not a differentiable training channel.

Executed: 11 deterministic numerical tests (source bits/CRC erasure, Gray alphabets, FEC impulse/loopback/noise correction, 8/14/16G clock/noiseless waveform, matched noise, actual4608-real FEC/CRC loopback, fixed-power rates, bandwidth impairment, strict report pairing) and dry-run144-row plan. Local Python3.12/NumPy2.3.5, no torch. Server environment must independently run both test files. Static Python3.8 grammar and shell syntax checks are part of release verification.

Current limitations: no source-codec JPEG/H264 baseline; same existing semantic checkpoint, no retraining; digital lacks token UEP; analog and digital times/energies differ; CRC failure whole-frame erasure/no ARQ; shared symbol clock, constant phase, no CFO/timing drift/shot-noise/polarization; ADC clock and optional bandwidth are uncalibrated assumptions. Scalar equalization is not CMA/DD-LMS/Volterra. No accuracy, publication, field feasibility or hardware throughput claim.

Next: install additively; CPU core and torch adapter checks, then bounded run_pilot.sh on physical GPU1. Defaults8validation images×3referenceSNR×2rates×3branches=144rows. New timestamp output, noAMP, unchanged mixed_seed42 checkpoint, same frozen split. Return all144 rows including CRC failures and image/caption outputs. report checks full condition lattice, input hashes and equal irradiance across rates. Physical-summary goodput excludes neural inference/actual device control delays; reverse RF64symbols at assumed1MHz separately charged.
