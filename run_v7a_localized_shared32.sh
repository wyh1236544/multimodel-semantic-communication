#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ROOT=/data1/zw/WYH/multimodel
cd "$ROOT"

# 10k/4-epoch screening. C32/I216/T40 and the old baseline both use 4608
# real channel symbols, so their result is a fair fixed-rate comparison.
python train.py \
  --config configs/v7a_localized_shared32.yaml \
  --preset original_4608 \
  --output-dir experiment_results/ln_v7a_shared32_seed42 \
  --init-checkpoint experiment_results/ln_v61_baseline_seed42/seed42.pth \
  --allow-shape-expansion \
  --seeds 42 \
  --epochs 4 \
  --train-samples 10000 \
  --val-samples 500 \
  --data-root "$ROOT/COCO2017" \
  --vae-path "$ROOT/pretrained/sd-vae-ft-mse" \
  --text-model-path "$ROOT/pretrained/clip-vit-base-patch32"

python test.py \
  --config configs/v7a_localized_shared32.yaml \
  --checkpoint experiment_results/ln_v7a_shared32_seed42/seed42.pth \
  --preset original_4608 \
  --output experiment_results/ln_v7a_shared32_seed42/eval.json \
  --snr-list 0 10 20 \
  --ablations full no_common common_only no_image_private no_text_private \
  --samples 500 \
  --seed 42 \
  --data-root "$ROOT/COCO2017" \
  --vae-path "$ROOT/pretrained/sd-vae-ft-mse" \
  --text-model-path "$ROOT/pretrained/clip-vit-base-patch32"
