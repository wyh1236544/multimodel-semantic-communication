#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

ROOT=/data1/zw/WYH/multimodel
OUTPUT="$ROOT/experiment_results/ln_v9a_feedback_generation_seed42"
SD_PATH="${SD_PATH:-$ROOT/pretrained/stable-diffusion-v1-5}"
CLIP_PATH="${CLIP_PATH:-$ROOT/pretrained/clip-vit-base-patch32}"
cd "$ROOT"
mkdir -p "$OUTPUT" logs

python -u check_v9_feedback_architecture.py

python -u train.py \
  --config configs/v9a_feedback_generation.yaml \
  --preset original_4608 \
  --output-dir "$OUTPUT" \
  --init-checkpoint experiment_results/ln_v8a_common_base_seed42/seed42.pth \
  --allow-shape-expansion \
  --seeds 42 \
  --epochs 6 \
  --train-samples 10000 \
  --val-samples 500 \
  --data-root "$ROOT/COCO2017" \
  --vae-path "$ROOT/pretrained/sd-vae-ft-mse" \
  --text-model-path "$CLIP_PATH"

# End-to-end feedback and task-switching panels. The received caption, never
# the source Localized Narrative, is used as the diffusion prompt.
python -u v9_feedback_task_adaptive_eval.py \
  --config configs/v9a_feedback_generation.yaml \
  --checkpoint "$OUTPUT/seed42.pth" \
  --output-dir "$OUTPUT/feedback_diffusion_panels" \
  --data-root "$ROOT/COCO2017" \
  --vae-path "$ROOT/pretrained/sd-vae-ft-mse" \
  --text-model-path "$CLIP_PATH" \
  --clip-model-path "$CLIP_PATH" \
  --diffusion-model-path "$SD_PATH" \
  --snr-list -5 0 5 10 20 \
  --samples 8 \
  --batch-size 4 \
  --steps 20 \
  --seed 42
