#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

ROOT=/data1/zw/WYH/multimodel
CHECKPOINT="$ROOT/experiment_results/ln_v9a_feedback_generation_seed42/seed42.pth"
OUTPUT="$ROOT/experiment_results/ln_v9a_feedback_generation_seed42/strength_sweep"
CLIP_PATH="$ROOT/pretrained/clip-vit-base-patch32"
SD_PATH="$ROOT/pretrained/stable-diffusion-v1-5"
cd "$ROOT"
mkdir -p "$OUTPUT" logs

# 20 paired validation images per SNR. Every strength reuses exactly the same
# communication realization and diffusion seed.
python -u v9_feedback_task_adaptive_eval.py \
  --config configs/v9a_feedback_generation.yaml \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --data-root "$ROOT/COCO2017" \
  --vae-path "$ROOT/pretrained/sd-vae-ft-mse" \
  --text-model-path "$CLIP_PATH" \
  --clip-model-path "$CLIP_PATH" \
  --diffusion-model-path "$SD_PATH" \
  --snr-list -5 0 5 \
  --samples 20 \
  --batch-size 4 \
  --steps 20 \
  --hybrid-strength-list 0.15 0.25 0.35 0.45 \
  --generation-strength-list 0.55 0.65 0.75 0.85 \
  --seed 42

python -u tools/select_v9_diffusion_strength.py \
  "$OUTPUT/summary.json" \
  --max-mean-psnr-drop 1.5 \
  | tee "$OUTPUT/strength_ranking.txt"
