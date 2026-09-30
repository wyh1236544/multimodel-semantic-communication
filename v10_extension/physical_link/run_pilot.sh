#!/usr/bin/env bash
set -euo pipefail
cd /data1/zw/WYH/multimodel
export CUDA_VISIBLE_DEVICES=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/data1/zw/WYH/multimodel/.cache/v10/huggingface
export TRANSFORMERS_CACHE=/data1/zw/WYH/multimodel/.cache/v10/huggingface/transformers
export TORCH_HOME=/data1/zw/WYH/multimodel/.cache/v10/torch
export TMPDIR=/data1/zw/WYH/multimodel/.cache/v10/tmp
mkdir -p "$TMPDIR" logs/v10
nvidia-smi -i 1 --query-gpu=index,name,memory.used,memory.free --format=csv
V2_RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
V2_OUTPUT="experiment_results/v10_physical_link_v2_${V2_RUN_TAG}"
test ! -e "$V2_OUTPUT"
/home/zw/anaconda3/envs/torch/bin/python -B -u v10_extension/physical_link/run.py \
  --checkpoint experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth \
  --split-manifest v10_extension/manifests/ln_v10_splits.json \
  --output-dir "$V2_OUTPUT" --data-root COCO2017 \
  --vae-path pretrained/sd-vae-ft-mse --text-model-path pretrained/clip-vit-base-patch32 \
  --variants analog 16qam_none 16qam_conv \
  --symbol-rates 8000000000 14000000000 --reference-symbol-rate 14000000000 \
  --sample-rate 64000000000 --rolloff 0.01 --span 128 \
  --profiles fso_moderate --snr-list 0 10 20 --samples 8 --repeats 1 --save-images 4 \
  --max-hours 0.5 --max-total-hours 2 \
  2>&1 | tee "logs/v10/physical_link_v2_${V2_RUN_TAG}.log"
echo "Output: $V2_OUTPUT"
