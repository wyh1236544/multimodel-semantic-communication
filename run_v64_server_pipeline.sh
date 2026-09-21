#!/usr/bin/env bash
set -euo pipefail

# Run after: cd /data1/zw/WYH/multimodel && conda activate torch
STAGE="${1:-help}"
GPU_ID="${GPU_ID:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-/data1/zw/WYH/multimodel}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/COCO2017}"
V61_CHECKPOINT="${V61_CHECKPOINT:-${PROJECT_ROOT}/V6.1_model_weights/seed42.pth}"
VAE_PATH="${VAE_PATH:-${PROJECT_ROOT}/pretrained/sd-vae-ft-mse}"
CLIP_PATH="${CLIP_PATH:-${PROJECT_ROOT}/pretrained/clip-vit-base-patch32}"
SD_PATH="${SD_PATH:-${PROJECT_ROOT}/pretrained/stable-diffusion-v1-5}"
CONTROLNET_PATH="${CONTROLNET_PATH:-${PROJECT_ROOT}/pretrained/control_v11f1e_sd15_tile}"
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:-${PROJECT_ROOT}/V6.2_model_weights/adapter/semantic_adapter_latest.pth}"
POLICY_DATA="${POLICY_DATA:-${PROJECT_ROOT}/experiment_results/v64_policy_data/policy_dataset.pth}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-${PROJECT_ROOT}/V6.4_model_weights/policy/controlnet_policy_best.pth}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/v64}"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

run_check() {
  PYTHONUNBUFFERED=1 python check_v64_policy_architecture.py \
    2>&1 | tee "${LOG_ROOT}/01_policy_architecture.log"
  PYTHONUNBUFFERED=1 python check_v63_controlnet_environment.py \
    --controlnet-path "${CONTROLNET_PATH}" --load-model \
    2>&1 | tee "${LOG_ROOT}/02_controlnet_environment.log"
}

run_collect() {
  test -f "${ADAPTER_CHECKPOINT}" || { echo "Missing adapter: ${ADAPTER_CHECKPOINT}" >&2; exit 2; }
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python collect_v64_controlnet_policy_data.py \
    --checkpoint "${V61_CHECKPOINT}" \
    --adapter-checkpoint "${ADAPTER_CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --vae-path "${VAE_PATH}" \
    --text-model-path "${CLIP_PATH}" \
    --diffusion-model-path "${SD_PATH}" \
    --controlnet-path "${CONTROLNET_PATH}" \
    --clip-model-path "${CLIP_PATH}" \
    --samples "${V64_POLICY_SAMPLES:-20}" \
    --output-dir experiment_results/v64_policy_data \
    2>&1 | tee "${LOG_ROOT}/03_policy_collect.log"
}

run_policy() {
  test -f "${POLICY_DATA}" || { echo "Missing policy data: ${POLICY_DATA}" >&2; exit 2; }
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python train_v64_controlnet_policy.py \
    --dataset "${POLICY_DATA}" \
    --epochs "${V64_POLICY_EPOCHS:-100}" \
    --output-dir V6.4_model_weights/policy \
    2>&1 | tee "${LOG_ROOT}/04_policy_train.log"
}

run_evaluate() {
  test -f "${POLICY_CHECKPOINT}" || { echo "Missing policy: ${POLICY_CHECKPOINT}" >&2; exit 2; }
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python v64_controlnet_adaptive_eval.py \
    --checkpoint "${V61_CHECKPOINT}" \
    --adapter-checkpoint "${ADAPTER_CHECKPOINT}" \
    --policy-checkpoint "${POLICY_CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --vae-path "${VAE_PATH}" \
    --text-model-path "${CLIP_PATH}" \
    --diffusion-model-path "${SD_PATH}" \
    --controlnet-path "${CONTROLNET_PATH}" \
    --clip-model-path "${CLIP_PATH}" \
    --samples "${V64_EVAL_SAMPLES:-12}" \
    --route-threshold "${V64_ROUTE_THRESHOLD:-0.5}" \
    --text-safety "${V64_TEXT_SAFETY:-heuristic_max}" \
    --seed "${V64_EVAL_SEED:-242}" \
    --output-dir "${V64_EVAL_OUTPUT:-experiment_results/v64_holdout_calibrated_seed242}" \
    2>&1 | tee "${LOG_ROOT}/05_holdout_eval.log"
}

case "${STAGE}" in
  check) run_check ;;
  collect) run_collect ;;
  policy) run_policy ;;
  evaluate) run_evaluate ;;
  quick) run_check; run_collect; run_policy; run_evaluate ;;
  *)
    echo "Usage: bash run_v64_server_pipeline.sh {check|collect|policy|evaluate|quick}"
    echo "Pilot collection: V64_POLICY_SAMPLES=20 bash run_v64_server_pipeline.sh collect"
    echo "Formal collection: V64_POLICY_SAMPLES=200 bash run_v64_server_pipeline.sh collect"
    ;;
esac
