#!/usr/bin/env bash
set -euo pipefail

# Run after: cd /data1/zw/WYH/multimodel && conda activate torch
# Override any value as an environment variable before this command.
STAGE="${1:-help}"
GPU_ID="${GPU_ID:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-/data1/zw/WYH/multimodel}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/COCO2017}"
V61_CHECKPOINT="${V61_CHECKPOINT:-${PROJECT_ROOT}/V6.1_model_weights/seed42.pth}"
VAE_PATH="${VAE_PATH:-${PROJECT_ROOT}/pretrained/sd-vae-ft-mse}"
CLIP_PATH="${CLIP_PATH:-${PROJECT_ROOT}/pretrained/clip-vit-base-patch32}"
SD_PATH="${SD_PATH:-${PROJECT_ROOT}/pretrained/stable-diffusion-v1-5}"
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:-${PROJECT_ROOT}/V6.2_model_weights/adapter/semantic_adapter_latest.pth}"
POLICY_DATA="${POLICY_DATA:-${PROJECT_ROOT}/experiment_results/v62_policy_data/policy_dataset.pth}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-${PROJECT_ROOT}/V6.2_model_weights/policy/quality_policy_best.pth}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/v62}"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

run_check() {
  python check_v62_architecture.py | tee "${LOG_ROOT}/01_architecture_check.log"
  python check_v62_diffusion_environment.py --load-pipeline \
    --diffusion-model-path "${SD_PATH}" \
    --clip-model-path "${CLIP_PATH}" \
    --vae-path "${VAE_PATH}" | tee "${LOG_ROOT}/02_diffusion_environment.log"
}

run_pilot() {
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python v62_quality_adaptive_eval.py \
    --checkpoint "${V61_CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --vae-path "${VAE_PATH}" \
    --text-model-path "${CLIP_PATH}" \
    --diffusion-model-path "${SD_PATH}" \
    --clip-model-path "${CLIP_PATH}" \
    --snr-list 0 10 20 \
    --samples "${V62_PILOT_SAMPLES:-12}" \
    --batch-size 2 \
    --output-dir experiment_results/v62_pilot \
    2>&1 | tee "${LOG_ROOT}/03_pilot.log"
}

run_adapter() {
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python train_v62_semantic_adapter.py \
    --checkpoint "${V61_CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --vae-path "${VAE_PATH}" \
    --text-model-path "${CLIP_PATH}" \
    --clip-model-path "${CLIP_PATH}" \
    --diffusion-model-path "${SD_PATH}" \
    --train-samples "${V62_ADAPTER_SAMPLES:-3000}" \
    --max-steps "${V62_ADAPTER_STEPS:-1000}" \
    --batch-size 1 \
    --output-dir V6.2_model_weights/adapter \
    2>&1 | tee "${LOG_ROOT}/04_adapter_train.log"
}

run_collect() {
  test -f "${ADAPTER_CHECKPOINT}" || {
    echo "Missing adapter checkpoint: ${ADAPTER_CHECKPOINT}" >&2
    exit 2
  }
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python collect_v62_policy_data.py \
    --checkpoint "${V61_CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --vae-path "${VAE_PATH}" \
    --text-model-path "${CLIP_PATH}" \
    --diffusion-model-path "${SD_PATH}" \
    --clip-model-path "${CLIP_PATH}" \
    --adapter-checkpoint "${ADAPTER_CHECKPOINT}" \
    --snr-list -5 0 5 10 15 20 \
    --samples "${V62_POLICY_SAMPLES:-50}" \
    --output-dir experiment_results/v62_policy_data \
    2>&1 | tee "${LOG_ROOT}/05_policy_collect.log"
}

run_policy() {
  test -f "${POLICY_DATA}" || {
    echo "Missing policy dataset: ${POLICY_DATA}" >&2
    exit 2
  }
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python train_v62_quality_policy.py \
    --dataset "${POLICY_DATA}" \
    --epochs "${V62_POLICY_EPOCHS:-80}" \
    --output-dir V6.2_model_weights/policy \
    2>&1 | tee "${LOG_ROOT}/06_policy_train.log"
}

run_evaluate() {
  test -f "${ADAPTER_CHECKPOINT}" || {
    echo "Missing adapter checkpoint: ${ADAPTER_CHECKPOINT}" >&2
    exit 2
  }
  test -f "${POLICY_CHECKPOINT}" || {
    echo "Missing policy checkpoint: ${POLICY_CHECKPOINT}" >&2
    exit 2
  }
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python v62_quality_adaptive_eval.py \
    --checkpoint "${V61_CHECKPOINT}" \
    --policy-checkpoint "${POLICY_CHECKPOINT}" \
    --adapter-checkpoint "${ADAPTER_CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --vae-path "${VAE_PATH}" \
    --text-model-path "${CLIP_PATH}" \
    --diffusion-model-path "${SD_PATH}" \
    --clip-model-path "${CLIP_PATH}" \
    --snr-list -5 0 5 10 15 20 \
    --samples "${V62_EVAL_SAMPLES:-20}" \
    --full-ablation \
    --output-dir experiment_results/v62_final \
    2>&1 | tee "${LOG_ROOT}/07_holdout_evaluation.log"
  python plot_v62_results.py \
    --summary experiment_results/v62_final/summary.csv \
    --output experiment_results/v62_final/v62_summary.png \
    2>&1 | tee "${LOG_ROOT}/08_plot.log"
}

case "${STAGE}" in
  check) run_check ;;
  pilot) run_pilot ;;
  adapter) run_adapter ;;
  collect) run_collect ;;
  policy) run_policy ;;
  evaluate) run_evaluate ;;
  quick)
    run_check
    run_pilot
    run_adapter
    run_collect
    run_policy
    run_evaluate
    ;;
  *)
    echo "Usage: bash run_v62_server_pipeline.sh {check|pilot|adapter|collect|policy|evaluate|quick}"
    echo "Examples:"
    echo "  bash run_v62_server_pipeline.sh check"
    echo "  GPU_ID=1 bash run_v62_server_pipeline.sh pilot"
    echo "  V62_ADAPTER_STEPS=3000 bash run_v62_server_pipeline.sh adapter"
    echo "  V62_POLICY_SAMPLES=500 bash run_v62_server_pipeline.sh collect"
    echo "  V62_EVAL_SAMPLES=500 bash run_v62_server_pipeline.sh evaluate"
    ;;
esac
