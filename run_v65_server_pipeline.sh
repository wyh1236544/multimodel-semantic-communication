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
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-${PROJECT_ROOT}/V6.4_model_weights/policy/controlnet_policy_best.pth}"
VERIFIER_DATA="${VERIFIER_DATA:-${PROJECT_ROOT}/experiment_results/v65_verifier_data/verifier_dataset.pth}"
VERIFIER_CHECKPOINT="${VERIFIER_CHECKPOINT:-${PROJECT_ROOT}/V6.5.1_model_weights/verifier/generation_verifier_best.pth}"
VERIFIER_THRESHOLDS="${VERIFIER_THRESHOLDS:-${PROJECT_ROOT}/V6.5.1_model_weights/verifier/verifier_thresholds.json}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/v651}"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

run_check() {
  PYTHONUNBUFFERED=1 python check_v65_architecture.py \
    2>&1 | tee "${LOG_ROOT}/01_architecture.log"
  PYTHONUNBUFFERED=1 python check_v63_controlnet_environment.py \
    --controlnet-path "${CONTROLNET_PATH}" --load-model \
    2>&1 | tee "${LOG_ROOT}/02_environment.log"
}

run_collect() {
  test -f "${POLICY_CHECKPOINT}" || { echo "Missing V6.4 policy: ${POLICY_CHECKPOINT}" >&2; exit 2; }
  test -f "${ADAPTER_CHECKPOINT}" || { echo "Missing adapter: ${ADAPTER_CHECKPOINT}" >&2; exit 2; }
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python collect_v65_verifier_data.py \
    --checkpoint "${V61_CHECKPOINT}" \
    --adapter-checkpoint "${ADAPTER_CHECKPOINT}" \
    --policy-checkpoint "${POLICY_CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --vae-path "${VAE_PATH}" \
    --text-model-path "${CLIP_PATH}" \
    --diffusion-model-path "${SD_PATH}" \
    --controlnet-path "${CONTROLNET_PATH}" \
    --clip-model-path "${CLIP_PATH}" \
    --samples "${V65_VERIFIER_SAMPLES:-100}" \
    --candidate-probability-floor "${V65_CANDIDATE_FLOOR:-0.30}" \
    --seed "${V65_COLLECT_SEED:-342}" \
    --output-dir "${V65_COLLECT_OUTPUT:-experiment_results/v65_verifier_data}" \
    2>&1 | tee "${LOG_ROOT}/03_verifier_collect.log"
}

run_train() {
  test -f "${VERIFIER_DATA}" || { echo "Missing verifier data: ${VERIFIER_DATA}" >&2; exit 2; }
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python train_v65_verifier.py \
    --dataset "${VERIFIER_DATA}" \
    --epochs "${V651_VERIFIER_EPOCHS:-140}" \
    --deployment-weight "${V651_DEPLOYMENT_WEIGHT:-2.0}" \
    --output-dir V6.5.1_model_weights/verifier \
    2>&1 | tee "${LOG_ROOT}/04_verifier_train.log"
}

run_inspect() {
  test -f "${VERIFIER_DATA}" || { echo "Missing verifier data: ${VERIFIER_DATA}" >&2; exit 2; }
  PYTHONUNBUFFERED=1 python inspect_v65_dataset.py \
    --dataset "${VERIFIER_DATA}" \
    2>&1 | tee "${LOG_ROOT}/03b_verifier_inspect.log"
}

run_calibrate() {
  test -f "${VERIFIER_CHECKPOINT}" || { echo "Missing verifier: ${VERIFIER_CHECKPOINT}" >&2; exit 2; }
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python calibrate_v65_verifier.py \
    --dataset "${VERIFIER_DATA}" \
    --verifier-checkpoint "${VERIFIER_CHECKPOINT}" \
    --output "${VERIFIER_THRESHOLDS}" \
    2>&1 | tee "${LOG_ROOT}/05_verifier_calibrate.log"
}

run_evaluate() {
  test -f "${VERIFIER_CHECKPOINT}" || { echo "Missing verifier: ${VERIFIER_CHECKPOINT}" >&2; exit 2; }
  test -f "${VERIFIER_THRESHOLDS}" || { echo "Missing thresholds: ${VERIFIER_THRESHOLDS}" >&2; exit 2; }
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python v65_generate_verify_eval.py \
    --checkpoint "${V61_CHECKPOINT}" \
    --adapter-checkpoint "${ADAPTER_CHECKPOINT}" \
    --policy-checkpoint "${POLICY_CHECKPOINT}" \
    --verifier-checkpoint "${VERIFIER_CHECKPOINT}" \
    --verifier-thresholds "${VERIFIER_THRESHOLDS}" \
    --data-root "${DATA_ROOT}" \
    --vae-path "${VAE_PATH}" \
    --text-model-path "${CLIP_PATH}" \
    --diffusion-model-path "${SD_PATH}" \
    --controlnet-path "${CONTROLNET_PATH}" \
    --clip-model-path "${CLIP_PATH}" \
    --samples "${V651_EVAL_SAMPLES:-12}" \
    --route-threshold "${V651_ROUTE_THRESHOLD:-0.70}" \
    --text-safety "${V651_TEXT_SAFETY:-hybrid}" \
    --text-selection-margin "${V651_TEXT_SELECTION_MARGIN:-0.005}" \
    --seed "${V651_EVAL_SEED:-642}" \
    --output-dir "${V651_EVAL_OUTPUT:-experiment_results/v651_holdout_seed642}" \
    2>&1 | tee "${LOG_ROOT}/06_holdout_eval.log"
}

case "${STAGE}" in
  check) run_check ;;
  collect) run_collect ;;
  inspect) run_inspect ;;
  train) run_train ;;
  calibrate) run_calibrate ;;
  evaluate) run_evaluate ;;
  quick) run_check; run_collect; run_train; run_calibrate; run_evaluate ;;
  *)
    echo "Usage: bash run_v65_server_pipeline.sh {check|collect|inspect|train|calibrate|evaluate|quick}"
    echo "Pilot: V65_VERIFIER_SAMPLES=40 bash run_v65_server_pipeline.sh collect"
    echo "Recommended: V65_VERIFIER_SAMPLES=100 bash run_v65_server_pipeline.sh collect"
    ;;
esac
