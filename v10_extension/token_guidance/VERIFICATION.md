# Local verification evidence

2026-10-01. Runtime: bundled Python3.12/NumPy on macOS; NOT the server
Python3.8/PyTorch1.12 runtime. Actual Torch CPU/GPU/COCO tests remain unexecuted
locally because the provided runtime has no PyTorch. No installation attempted.

Executed, exit0:

- token_guidance/test_core.py:5 tests passed. Actual8-bit semantic-object
  quantization, no-noise16QAM/FEC/RRC/DAC delivery, noisy CRC erasure, no UTF8 wire
  task, donor nonidentity and incomplete/duplicate/unpaired/nonfinite rejection.
- business_link/test_core.py:10 unchanged regression tests passed.
- physical_link/test_core.py:11 unchanged regression tests passed.
- All12 new Python files parse with Python3.8 grammar.

A preliminary command targeted nonexistent physical_link/test_physical.py and
returned file-not-found; corrected to test_core.py above, which passed. This was
a verification-command filename error, not a successful or failed model run.

Release builder verifies archive member allowlist, extracted source SHA256,
additive installation, identical reinstall, and rejection/preservation of a
conflicting user file in isolated project-local synthetic directories. Builder
must exit0 and output the archive SHA before a release is considered packaged.

Still required on server: launch.py check (includes3 Torch hook/STE checks),
four-arm8-image smoke with enabled-guidance nonzero gradients/updates and strict
real checkpoint loading; then the paired quick COCO validation. Numeric code
tests and packaging do NOT demonstrate image/text quality improvement.
