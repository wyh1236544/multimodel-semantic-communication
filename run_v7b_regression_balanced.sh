#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ROOT=/data1/zw/WYH/multimodel
cd "$ROOT"

python train.py \
  --config configs/v7b_regression_balanced.yaml \
  --preset original_4608 \
  --output-dir experiment_results/ln_v7b_regression_balanced_seed42 \
  --init-checkpoint experiment_results/ln_v7a_shared32_seed42/seed42.pth \
  --allow-shape-expansion \
  --seeds 42 \
  --epochs 4 \
  --train-samples 10000 \
  --val-samples 500 \
  --data-root "$ROOT/COCO2017" \
  --vae-path "$ROOT/pretrained/sd-vae-ft-mse" \
  --text-model-path "$ROOT/pretrained/clip-vit-base-patch32"

python test.py \
  --config configs/v7b_regression_balanced.yaml \
  --checkpoint experiment_results/ln_v7b_regression_balanced_seed42/seed42.pth \
  --preset original_4608 \
  --output experiment_results/ln_v7b_regression_balanced_seed42/eval.json \
  --snr-list 0 10 20 \
  --ablations full no_common common_only no_image_private no_text_private \
  --samples 500 \
  --seed 42 \
  --data-root "$ROOT/COCO2017" \
  --vae-path "$ROOT/pretrained/sd-vae-ft-mse" \
  --text-model-path "$ROOT/pretrained/clip-vit-base-patch32"

# Paired inference ablation: identical weights/noise, but receiver SNR gate off.
python test.py \
  --config configs/v7b_regression_balanced.yaml \
  --checkpoint experiment_results/ln_v7b_regression_balanced_seed42/seed42.pth \
  --preset original_4608 \
  --output experiment_results/ln_v7b_regression_balanced_seed42/eval_no_text_snr_gate.json \
  --snr-list 0 10 20 \
  --ablations full \
  --samples 500 \
  --seed 42 \
  --disable-text-snr-gate \
  --data-root "$ROOT/COCO2017" \
  --vae-path "$ROOT/pretrained/sd-vae-ft-mse" \
  --text-model-path "$ROOT/pretrained/clip-vit-base-patch32"
