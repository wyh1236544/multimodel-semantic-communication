#!/usr/bin/env bash
set -euo pipefail

# Run after: cd /data1/zw/WYH/multimodel && conda activate torch
GPU_ID="${GPU_ID:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-/data1/zw/WYH/multimodel}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/COCO2017}"
V61_CHECKPOINT="${V61_CHECKPOINT:-${PROJECT_ROOT}/V6.1_model_weights/seed42.pth}"
VAE_PATH="${VAE_PATH:-${PROJECT_ROOT}/pretrained/sd-vae-ft-mse}"
CLIP_PATH="${CLIP_PATH:-${PROJECT_ROOT}/pretrained/clip-vit-base-patch32}"
SD_PATH="${SD_PATH:-${PROJECT_ROOT}/pretrained/stable-diffusion-v1-5}"
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:-${PROJECT_ROOT}/V6.2_model_weights/adapter/semantic_adapter_latest.pth}"
CONTROLNET_PATH="${CONTROLNET_PATH:-${PROJECT_ROOT}/pretrained/control_v11f1e_sd15_tile}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/v63}"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

STAGE="${1:-help}"
case "${STAGE}" in
  check)
    PYTHONUNBUFFERED=1 python check_v63_controlnet_environment.py \
      --controlnet-path "${CONTROLNET_PATH}" \
      --load-model \
      2>&1 | tee "${LOG_ROOT}/01_controlnet_check.log"
    ;;
  pilot)
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      python v63_controlnet_fusion_pilot.py \
      --checkpoint "${V61_CHECKPOINT}" \
      --adapter-checkpoint "${ADAPTER_CHECKPOINT}" \
      --data-root "${DATA_ROOT}" \
      --vae-path "${VAE_PATH}" \
      --text-model-path "${CLIP_PATH}" \
      --diffusion-model-path "${SD_PATH}" \
      --controlnet-path "${CONTROLNET_PATH}" \
      --clip-model-path "${CLIP_PATH}" \
      --snr-list 0 10 \
      --samples "${V63_SAMPLES:-6}" \
      --output-dir experiment_results/v63_controlnet_pilot \
      2>&1 | tee "${LOG_ROOT}/02_controlnet_pilot.log"
    ;;
  *)
    echo "Usage: bash run_v63_controlnet_pilot.sh {check|pilot}"
    ;;
esac
