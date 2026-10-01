# Pilot gain V1 handoff — 2026-10-01

User requested supplementary experiment code to measure pilot-driven adaptation gains.
Additive package only; four same-checkpoint validation policies, original continuous RF/IMDD.
No source/weights/results/split/global environment edits; no GPU/server job started here.
Source folder contains runner, instrumented evaluator wrapper, strict paired report and stdlib tests.
Pilot/gain and payload RNG fingerprints are diagnostics only, never receiver side information.
Read README.md for exact scope, run instructions and interpretation limits.
Local synthetic/static verification is separate from pending real torch1.12/COCO/GPU smoke.
Local: 10 stdlib tests passed, Python3.8 grammar and 9600-row dry-run checked.
Actual numerical channel test skipped locally because PyTorch is unavailable; run on server.
