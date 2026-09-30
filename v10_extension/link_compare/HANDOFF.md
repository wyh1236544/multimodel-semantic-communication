# Link-compare v1 handoff

Status: code delivered for a validation-only exploratory pilot; **no server run or
measured improvement has been claimed**. This is an additive module. Existing
V10 source, checkpoints, manifests, results, and the shared handoff are untouched.

Run the CPU-independent `test_rate_report.py` and the PyTorch-backed
`test_link_compare.py` in the existing server environment before the pilot.
Use the exact GPU-1 pilot command in `README.md` after checking GPU ownership
and free memory. `CUDA_VISIBLE_DEVICES=1` maps physical GPU 1 to `cuda:0`
inside the process. Do not terminate or use another person's GPU process.

The pilot evaluates one unchanged V10 mixed checkpoint on validation images,
contrasting original IM/DD continuous slots with pilot-aided coherent complex
symbols and zero-shot QPSK/16-QAM hard constellation constraints. Two **assumed**
symbol rates form a fixed-power/white-noise normalized sensitivity sweep. There
is no measured link throughput, matched optical power, codec, FEC, waveform, or
hardware validation. Do not frame the eight-image pilot as paper evidence.

Outputs: `design.json`, per-variant/rate `summary.json`, `samples.jsonl`, image
examples, `COMPLETE.json`, then root `rate_summary.json`, CSV, and `COMPLETE.json`.
If a run fails, preserve the partial output and use a fresh directory for retry.
Next action after a completed run: inspect all conditions and image/text examples,
then decide whether to train a discrete-aware model and build a matched-power
codec/FEC baseline before any test split or hardware claim.
