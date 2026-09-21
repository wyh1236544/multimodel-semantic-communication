#!/usr/bin/env bash
set -euo pipefail
# Execute from the existing project root; never cd to a sibling project.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-/home/zw/anaconda3/envs/torch/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python missing: %s\nSet PYTHON_BIN to your existing torch environment.\n' "$PYTHON_BIN" >&2
  exit 2
fi
exec "$PYTHON_BIN" -B -u v10_extension/launch_v10.py "$@"
