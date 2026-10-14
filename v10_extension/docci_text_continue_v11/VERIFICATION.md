# Verification boundaries

Local runtime Python3.12 + NumPy/Pillow, no Torch/Transformers. All new Python is
parsed with Python3.8 grammar. Local core/report tests cover exact edit distance,
legacy metrics, correct full62/EOS classification, partial early EOS, global
cursor and20epoch schedule, complete/duplicate/missing/source-leak/test/metric/
channel/EOS-negative matrices and privileged prefix-diagnostic scope.

Release builder executes source/restore/install/idempotence/conflict/drift/symlink/
divergent-HANDOFF/byte-backup checks and repeats core/report tests in an isolated
project. Source archive is USTAR, no macOS xattrs, weights/data/cache/results.
Real prior2048-record metrics/source-length regression is recorded separately
in the build verification, not mislabeled neural re-inference.

Mandatory server3 real Torch/HF tests cover every codec group update/frozen BART,
actual saved model+optimizer/RNG roundtrip and parent-state restoration, unchanged
public receiver signature, free/true-prefix diagnostic immutability. Inherited
10 Torch/HF V10 tests remain mandatory. Authentic exact parent/BART/GPU smoke is
executed before main in the production entry, never reused for training init.

Actual saved pretrained checkpoint loading, GPU smoke/FP32 memory, continuation
speed and quality gain are NOT_EXECUTED locally. No remote task or environment
changes performed. One fixed seed, reused qual_dev; no independent test,
scheduled sampling, diffusion, CLIP/Common, FSO or digital superiority evidence.
New AdamW phase is explicitly NOT restoration of the absent parent optimizer.
Legacy quality gates are unchanged; full source/framing/failure records retained.
