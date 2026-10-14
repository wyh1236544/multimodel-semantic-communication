## 2026-10-14 V11 CPU-only immutable collection recovery

User's completed run v10_docci_text_continue_v11_20261014_201245 reached all100 validation
and RUN_FINISHED, then collector rejected prefix FP32 weighted aggregates. Read-only
user audit found31 strict1e-7 failures; five printed errors5.83e-8..3.32e-7 support
rounding mismatch. Only those five scalar examples are locally available, not all31
raw rows. Never claim all raw records passed before the server reruns full verification.

New additive docci_text_collect_fix_v11 only. Old V11 code/MANIFEST/source_snapshot,
diagnostic values, data, checkpoints, failed output and logs remain byte-preserved.
No Torch import, pickle loading, GPU, training or generation. Only diagnostic function
is scoped to a declared new verifier; every nonaggregation validator AST matches old,
other comparisons/quality WER/EOS/margin gates unchanged and restored even on failure.

Finite nonnegative original FP32 means, exact public integer counts<=510/slot<=62;
error allowance computed from FP32 reduction roundoff bounds, not observed fitted
threshold. Full raw/hash/frame/matrix/history/saved-checkpoint checks still execute.
Actual differences, counts, allowance and prior-strict rejection flags retained in audit.
Excess discrepancy/hash/wire/count/scope/nonfinite failures remain fatal.

Full verify original first, copy only light records with before/after SHA checks into
NEW separate experiment_results/v10_docci_text_continue_v11_20261014_201245_recollect02,
then same legacy summary/HTML and new light archive. Original run gets no rewritten
COMPLETE/summary or other writes. Source fix snapshot and explicit same-neural-run
receipt archived. Partial output preserved; use recollect03, never overwrite retry.

Existing offline Python3.8/NumPy only; no dependency installation/network/new model.
16 local numeric/structural/copy tests plus old19 CPU regressions and isolated install/
source/USTAR/restore conflicts; real raw31/all-run server verification NOT_EXECUTED
locally. CPU collection cap30min is not ETA. Source tar/checksum and atomic GitHub
save required. See module README; no server task launched by delivery.
