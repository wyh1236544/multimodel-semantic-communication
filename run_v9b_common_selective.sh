#!/usr/bin/env bash
set -euo pipefail

ROOT="${V9B_ROOT:-/data1/zw/WYH/multimodel}"
CLIP_PATH="${V9B_CLIP_PATH:-$ROOT/pretrained/clip-vit-base-patch32}"
VAE_PATH="${V9B_VAE_PATH:-$ROOT/pretrained/sd-vae-ft-mse}"
SD_PATH="${V9B_SD_PATH:-$ROOT/pretrained/stable-diffusion-v1-5}"
BASE_CHECKPOINT="${V9B_BASE_CHECKPOINT:-$ROOT/experiment_results/ln_v9a_feedback_generation_seed42/seed42.pth}"
DATA_OUTPUT="${V9B_DATA_OUTPUT:-$ROOT/experiment_results/v9b2_verifier_data}"
WEIGHT_OUTPUT="${V9B_WEIGHT_OUTPUT:-$ROOT/V9.0B.2_model_weights/common_verifier}"
EVAL_OUTPUT="${V9B_EVAL_OUTPUT:-$ROOT/experiment_results/v9b2_common_selective_eval}"
STAGE="${1:-all}"
COLLECT_SEED="${V9B_COLLECT_SEED:-342}"
TRAIN_SEED="${V9B_TRAIN_SEED:-42}"
EVAL_SEED="${V9B_EVAL_SEED:-42}"
TRAIN_EXTRA=()
if [[ "${V9B_ALLOW_SMALL_DATASET:-0}" == "1" ]]; then
  TRAIN_EXTRA+=(--allow-small-dataset --min-train-per-class 1 --min-val-per-class 1)
fi

cd "$ROOT"
mkdir -p logs "$DATA_OUTPUT" "$WEIGHT_OUTPUT" "$EVAL_OUTPUT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

python -u check_v9b_architecture.py

if [[ "$STAGE" == "collect" || "$STAGE" == "all" ]]; then
  python -u collect_v9b_verifier_data.py \
    --config configs/v9a_feedback_generation.yaml \
    --checkpoint "$BASE_CHECKPOINT" \
    --output-dir "$DATA_OUTPUT" \
    --data-root "$ROOT/COCO2017" \
    --vae-path "$VAE_PATH" \
    --text-model-path "$CLIP_PATH" \
    --clip-model-path "$CLIP_PATH" \
    --diffusion-model-path "$SD_PATH" \
    --snr-list -5 0 5 \
    --samples "${V9B_COLLECT_SAMPLES:-300}" \
    --batch-size 4 --steps 20 \
    --generation-strength 0.55 --hybrid-strength 0.35 \
    --data-split train --seed "$COLLECT_SEED" \
    2>&1 | tee logs/v9b2_collect.log
fi

if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
  python -u train_v9b_common_verifier.py \
    --dataset "$DATA_OUTPUT/v9b_verifier_dataset.pt" \
    --output-dir "$WEIGHT_OUTPUT" \
    --epochs "${V9B_EPOCHS:-100}" \
    --batch-size 128 --common-dropout 0.25 \
    --generation-max-psnr-drop 1.0 --hybrid-max-psnr-drop 0.75 \
    --generation-max-ssim-drop 0.10 --hybrid-max-ssim-drop 0.08 \
    "${TRAIN_EXTRA[@]}" \
    --seed "$TRAIN_SEED" \
    2>&1 | tee logs/v9b2_train.log
fi

if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
  python -u v9b_common_selective_eval.py \
    --config configs/v9a_feedback_generation.yaml \
    --checkpoint "$BASE_CHECKPOINT" \
    --verifier-checkpoint "$WEIGHT_OUTPUT/v9b_verifier_best.pth" \
    --output-dir "$EVAL_OUTPUT" \
    --data-root "$ROOT/COCO2017" \
    --vae-path "$VAE_PATH" \
    --text-model-path "$CLIP_PATH" \
    --clip-model-path "$CLIP_PATH" \
    --diffusion-model-path "$SD_PATH" \
    --snr-list -5 0 5 10 20 \
    --samples "${V9B_EVAL_SAMPLES:-200}" \
    --batch-size 4 --steps 20 \
    --generation-strength 0.55 --hybrid-strength 0.35 \
    --data-split val --seed "$EVAL_SEED" \
    2>&1 | tee logs/v9b2_eval.log
fi
