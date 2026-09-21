#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ROOT=/data1/zw/WYH/multimodel
cd "$ROOT"

python train.py \
  --config configs/v8a_common_base.yaml \
  --preset original_4608 \
  --output-dir experiment_results/ln_v8a_common_base_seed42 \
  --init-checkpoint experiment_results/ln_v7b_regression_balanced_seed42/seed42.pth \
  --allow-shape-expansion \
  --seeds 42 \
  --epochs 6 \
  --train-samples 10000 \
  --val-samples 500 \
  --data-root "$ROOT/COCO2017" \
  --vae-path "$ROOT/pretrained/sd-vae-ft-mse" \
  --text-model-path "$ROOT/pretrained/clip-vit-base-patch32"

python test.py \
  --config configs/v8a_common_base.yaml \
  --checkpoint experiment_results/ln_v8a_common_base_seed42/seed42.pth \
  --preset original_4608 \
  --output experiment_results/ln_v8a_common_base_seed42/eval.json \
  --snr-list -5 0 5 10 20 \
  --ablations full common_only common_only_shuffled common_only_random no_common no_image_private no_text_private \
  --samples 500 \
  --seed 42 \
  --data-root "$ROOT/COCO2017" \
  --vae-path "$ROOT/pretrained/sd-vae-ft-mse" \
  --text-model-path "$ROOT/pretrained/clip-vit-base-patch32"

python visualize_recovery.py \
  --config configs/v8a_common_base.yaml \
  --checkpoint experiment_results/ln_v8a_common_base_seed42/seed42.pth \
  --preset original_4608 \
  --output-dir experiment_results/ln_v8a_common_base_seed42/common_only_panels \
  --semantic-ablation common_only \
  --snr-list -5 0 5 10 \
  --samples 16 \
  --batch-size 4 \
  --image-damage 0.0 \
  --seed 42 \
  --data-root "$ROOT/COCO2017" \
  --vae-path "$ROOT/pretrained/sd-vae-ft-mse" \
  --text-model-path "$ROOT/pretrained/clip-vit-base-patch32"
