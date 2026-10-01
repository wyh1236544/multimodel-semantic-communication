# Business-link V3 handoff

2026-10-01, additive evaluation-only extension. User authorized rewriting the
communication experiment to use fixed original image/text workload, 64GSa/s
AWG, 4sps, realistic packet/FEC/QAM/pulse-shaping accounting and received-text
assistance. No training, server run, environment upgrade, original-code write,
weight/result overwrite or external-device configuration is part of this turn.

Canonical operating guide and assumptions: [README.md](README.md).

Implemented: deterministic fixed validation task manifest; complete UTF8 text
wire task alongside unchanged joint semantic source encoding; quantized semantic
digital packet transport and continuous-feature comparison; Gray QPSK/16QAM;
actual CRC32/K7 rate1/2 soft Viterbi/repetition2, RRC/DAC64GSa/s4sps; same FSO
field shared by packets/modalities/retries; fixed modes or pilot/noisy-RF/optical
CRC-FEC mode handshake; explicit optional ideal-RF-ACK ARQ; delay/overhead/energy;
weighted pre/post BER, PER, EVM, pilot SNR, exact text/CER/WER, PSNR/SSIM; all
failure rows; strict lattice and field/source-payload pairing; zero-shot optional
received-text fusion versus same-wire baseline; full-condition constellation
and image exports; additive conflict-rejecting installer/source manifest,
deterministic source-only archive with extraction/install verification.

Defaults are engineering assumptions, not optimized/calibrated settings. New
fusion is not retrained or guaranteed to align with decoder feature space.
There is NO LDPC, automatic FEC controller, traditional image codec baseline,
diffusion execution or calibrated laser/device/link-budget model. Analog feature
BER/CRC and byte compression remain undefined rather than fabricated.

Local verification: NumPy/stdlib core suite passed 10 tests including actual UTF8
and quantized-feature waveform loops, all four digital policies, CRC erasure,
charged bounded failure retries, redundancy/time tradeoff, whole matrix and
bit-weighted aggregation and business-manifest reuse/input mismatch rejection.
The existing V2 NumPy suite passed all11 regression tests. Torch adapter and actual model/GPU/COCO tests require
the existing server torch environment; not claimed as executed locally.

Next: install inside existing /data1/zw/WYH/multimodel, run both CPU suites, then
8-forward GPU smoke on physical GPU1; after success, first compare8 images (384
forwards,2h maximum). Optional mode/retry runs are separate user-started new
outputs with reused manifest. Preserve failure evidence; do not launch training
or kill other users' processes. Full requested matrix + finite results + pairing,
not a cherry-picked quality improvement, is the runtime completion criterion.
