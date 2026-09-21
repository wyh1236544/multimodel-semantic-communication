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
CANNY_CONTROLNET_PATH="${CANNY_CONTROLNET_PATH:-${PROJECT_ROOT}/pretrained/control_v11p_sd15_canny}"
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:-${PROJECT_ROOT}/V6.2_model_weights/adapter/semantic_adapter_latest.pth}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-${PROJECT_ROOT}/V6.4_model_weights/policy/controlnet_policy_best.pth}"
VERIFIER_CHECKPOINT="${VERIFIER_CHECKPOINT:-${PROJECT_ROOT}/V6.5.1_model_weights/verifier/generation_verifier_best.pth}"
PREVIOUS_HOLDOUT="${PREVIOUS_HOLDOUT:-${PROJECT_ROOT}/experiment_results/v651_holdout_seed642/results.csv}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/v66}"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

run_check() {
  PYTHONUNBUFFERED=1 python check_v66_structure_environment.py \
    --canny-controlnet-path "${CANNY_CONTROLNET_PATH}" --load-model \
    2>&1 | tee "${LOG_ROOT}/01_environment.log"
}

run_oracle() {
  test -f "${VERIFIER_CHECKPOINT}" || { echo "Missing V6.5.1 verifier: ${VERIFIER_CHECKPOINT}" >&2; exit 2; }
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python v66_structure_sideinfo_pilot.py \
    --checkpoint "${V61_CHECKPOINT}" \
    --adapter-checkpoint "${ADAPTER_CHECKPOINT}" \
    --policy-checkpoint "${POLICY_CHECKPOINT}" \
    --verifier-checkpoint "${VERIFIER_CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --vae-path "${VAE_PATH}" \
    --text-model-path "${CLIP_PATH}" \
    --diffusion-model-path "${SD_PATH}" \
    --canny-controlnet-path "${CANNY_CONTROLNET_PATH}" \
    --clip-model-path "${CLIP_PATH}" \
    --structure-modes receiver oracle \
    --structure-side "${V66_STRUCTURE_SIDE:-64}" \
    --samples "${V66_ORACLE_SAMPLES:-6}" \
    --snr-list ${V66_ORACLE_SNRS:--5 0 5} \
    --strengths ${V66_STRENGTHS:-0.30 0.50} \
    --blend-alphas ${V66_BLEND_ALPHAS:-0.25 0.50} \
    --seed "${V66_ORACLE_SEED:-842}" \
    --exclude-results-csv "${PREVIOUS_HOLDOUT}" \
    --output-dir "${V66_ORACLE_OUTPUT:-experiment_results/v66_structure_oracle_seed842}" \
    2>&1 | tee "${LOG_ROOT}/02_oracle_pilot.log"
}

run_channel() {
  test -f "${VERIFIER_CHECKPOINT}" || { echo "Missing V6.5.1 verifier: ${VERIFIER_CHECKPOINT}" >&2; exit 2; }
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python v66_structure_sideinfo_pilot.py \
    --checkpoint "${V61_CHECKPOINT}" \
    --adapter-checkpoint "${ADAPTER_CHECKPOINT}" \
    --policy-checkpoint "${POLICY_CHECKPOINT}" \
    --verifier-checkpoint "${VERIFIER_CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --vae-path "${VAE_PATH}" \
    --text-model-path "${CLIP_PATH}" \
    --diffusion-model-path "${SD_PATH}" \
    --canny-controlnet-path "${CANNY_CONTROLNET_PATH}" \
    --clip-model-path "${CLIP_PATH}" \
    --structure-modes receiver oracle noisy \
    --structure-side "${V66_STRUCTURE_SIDE:-64}" \
    --structure-repetition "${V66_STRUCTURE_REPETITION:-0}" \
    --samples "${V66_CHANNEL_SAMPLES:-12}" \
    --snr-list ${V66_CHANNEL_SNRS:--5 0 5 10} \
    --strengths ${V66_STRENGTHS:-0.30 0.50} \
    --blend-alphas ${V66_BLEND_ALPHAS:-0.25 0.50} \
    --seed "${V66_CHANNEL_SEED:-942}" \
    --exclude-results-csv "${PREVIOUS_HOLDOUT}" \
    --output-dir "${V66_CHANNEL_OUTPUT:-experiment_results/v66_structure_channel_seed942}" \
    2>&1 | tee "${LOG_ROOT}/03_channel_pilot.log"
}

run_inspect() {
  SUMMARY="${V66_SUMMARY:-${PROJECT_ROOT}/experiment_results/v66_structure_oracle_seed842/summary.csv}"
  test -f "${SUMMARY}" || { echo "Missing V6.6 summary: ${SUMMARY}" >&2; exit 2; }
  PYTHONUNBUFFERED=1 python inspect_v66_results.py \
    --summary "${SUMMARY}" \
    2>&1 | tee "${LOG_ROOT}/02b_oracle_inspect.log"
}

case "${STAGE}" in
  check) run_check ;;
  oracle) run_oracle ;;
  inspect) run_inspect ;;
  channel) run_channel ;;
  pilot) run_check; run_oracle; run_inspect ;;
  *)
    echo "Usage: bash run_v66_server_pipeline.sh {check|oracle|inspect|channel|pilot}"
    echo "First decision gate: bash run_v66_server_pipeline.sh oracle"
    echo "Only run channel after the error-free oracle structure beats receiver-derived structure."
    ;;
esac
