# Verification boundary — 2026-10-13

Local host: macOS, bundled Python3.12, NumPy. No Torch/Transformers, real
pretrained files, COCO source data or school GPU/server access in this task.
No dependency installation or server task launched. Local checks cannot prove
the language codec learns, GPU memory fits, or text recovery improves.

## Executed locally

- New test_core.py: 10 non-Torch tests, including three complete compact V3
  waveform resource/shape cases, fixed SNR-before-fade noise, clean identity,
  no test source selection, strict configuration and no constrained-decode gate.
- New test_report.py: 11 synthetic verifier/negative tests, including negative
  native and validation stops, blocked FSO after failed validation, metrics,
  missing rows/updates, paired initialization, hash drift and archive roundtrip.
  Synthetic report fixtures do not represent any trained model's results.
- Old TextV3: test_core.py16 + test_report.py7 unchanged regressions passed.
- Old COCO modal V2: test_core.py11 + test_report.py10 unchanged regressions passed.
- Python3.8 grammar parsing of every new source. Launcher --help/no execution.
- Release builder enforces exact USTAR extraction/file hashes, install/reinstall,
  byte-exact shared handoff backup, old-source hashes unchanged, conflicting
  user file, prerequisite drift and symlink refusal. Actual build result is
  saved outside source archive as releases/v10_bart_latent_codec_v4_build_verification.json.

Exact commands use:
`/Users/a1-6/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B`
from the local project root. No local neural inference/training was executed.
The normal tar extraction deprecation warning under Python3.12 is not a test
failure; release tar contains only validated regular files and safe known names.

## Mandatory server-only checks

test_model.py10 Torch/HF tiny-model tests, then actual saved BART smoke:

- CE alone supplies nonzero gradients and real parameter updates in compression,
  TX normalization, reconstruction and memory groups; original BART unchanged.
- Fixed32 memory/public mask, source padding invariance, per-slot unit power.
- Teacher detached, padded targets ignored, official EOS/label shift.
- Same seed/init, restore, grad/no-grad numerics and erased-frame zero memory.
- Differentiable symbol proxy matches NumPy waveform-independent channel math.
- Actual cache vs encoder/incremental BART logits, disk checkpoint reload,
  real peak allocated/reserved GPU memory and all decoded texts.

These tests are NOT executed locally and NOT silently skipped by the launcher.
Real diagnostic uses2 arms/TRAIN8/1200updates each; no memory/quality assertion
is derived from parsing or synthetic files. SCREEN_COMPLETE means execution;
quality negative stops remain valid results, not publication evidence.
