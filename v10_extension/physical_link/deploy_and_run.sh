#!/usr/bin/env bash
# Used after uploading the release archive AND its .sha256 into multimodel.
set -euo pipefail
cd /data1/zw/WYH/multimodel
sha256sum -c v10_physical_link_v2.tar.gz.sha256
test ! -e v10_extension/physical_link
V2_STAGE="$(mktemp -d /data1/zw/WYH/multimodel/.v10_link_v2_stage.XXXXXX)"
tar -xzf v10_physical_link_v2.tar.gz -C "$V2_STAGE"
/home/zw/anaconda3/envs/torch/bin/python -B \
  "$V2_STAGE/v10_extension/physical_link/install.py" --project-root /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/physical_link/test_core.py
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/physical_link/test_adapter.py
echo 'Both CPU suites passed. Start GPU pilot separately: bash v10_extension/physical_link/run_pilot.sh'
# Keep stage/backup for recovery; no recursive deletion.
