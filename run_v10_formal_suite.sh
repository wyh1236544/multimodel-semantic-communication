#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PHASE="${1:-help}"
TAG="${FORMAL_TAG:-}"
SEEDS=(42 123 2026)
TRAIN=(--epochs 6 --train-samples 10000 --val-samples 500 --batch-size 2 --grad-accum 2 --num-workers 0 --max-hours 12)
EVAL_VAL=(--split validation --samples 500 --repeats 5 --save-images 12)
EVAL_TEST=(--split test --samples 1000 --repeats 5 --save-images 20)

if [[ "$PHASE" == help || -z "$TAG" ]]; then
  printf '%s\n' \
    'Usage: FORMAL_TAG=paper_v10_01 CUDA_VISIBLE_DEVICES=0 bash run_v10_formal_suite.sh PHASE' \
    'Phases: preflight, train-core, train-ablation, eval-validation, freeze, eval-test, aggregate' \
    'Run one phase at a time. test is locked until the freeze phase records the protocol.'
  exit $([[ "$PHASE" == help ]] && echo 0 || echo 2)
fi
if [[ ! "$TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'FORMAL_TAG contains unsafe characters: %s\n' "$TAG" >&2; exit 2
fi
ROOT="experiment_results/v10_formal_${TAG}"
FROZEN="v10_extension/manifests/formal_${TAG}_FROZEN.json"
mkdir -p "$ROOT"

train_one() {
  local seed="$1" name="$2"; shift 2
  bash run_v10.sh train "$ROOT/train_${name}_seed${seed}" "${TRAIN[@]}" --seed "$seed" "$@"
}
eval_one() {
  local seed="$1" name="$2" split="$3"; shift 3
  local budget=("${EVAL_VAL[@]}"); [[ "$split" == test ]] && budget=("${EVAL_TEST[@]}")
  bash run_v10.sh eval "$ROOT/train_${name}_seed${seed}/best.pth" \
    "$ROOT/eval_${split}_${name}_seed${seed}" "${budget[@]}" --seed "$seed" "$@"
}

case "$PHASE" in
  preflight)
    bash run_v10.sh check
    bash run_v10.sh splits
    bash run_v10.sh smoke "$ROOT/smoke_seed42" --batch-size 2 --num-workers 0 --seed 42
    ;;
  train-core)
    for seed in "${SEEDS[@]}"; do
      train_one "$seed" mixed
      train_one "$seed" awgn --regime awgn
      train_one "$seed" fso --regime fso_moderate
    done
    ;;
  train-ablation)
    for seed in "${SEEDS[@]}"; do
      train_one "$seed" no_uep --no-uep
      train_one "$seed" no_adapter --no-adapter
      train_one "$seed" no_common_probe --no-common-probe
    done
    ;;
  eval-validation)
    for seed in "${SEEDS[@]}"; do
      eval_one "$seed" mixed validation
      eval_one "$seed" awgn validation
      eval_one "$seed" fso validation
      eval_one "$seed" no_uep validation --no-uep
      eval_one "$seed" no_adapter validation --no-adapter
      eval_one "$seed" no_common_probe validation
      for ablation in common_only common_only_shuffled common_only_random; do
        bash run_v10.sh eval "$ROOT/train_mixed_seed${seed}/best.pth" \
          "$ROOT/eval_validation_${ablation}_seed${seed}" "${EVAL_VAL[@]}" \
          --seed "$seed" --ablation "$ablation"
      done
    done
    ;;
  freeze)
    [[ ! -e "$FROZEN" ]] || { printf 'Freeze record already exists: %s\n' "$FROZEN" >&2; exit 3; }
    for seed in "${SEEDS[@]}"; do
      [[ -f "$ROOT/eval_validation_mixed_seed${seed}/COMPLETE.json" ]] || { printf 'Missing validation seed %s\n' "$seed" >&2; exit 3; }
    done
    printf '{"tag":"%s","seeds":[42,123,2026],"test_policy":"single frozen evaluation; no tuning after test","status":"frozen"}\n' "$TAG" > "$FROZEN"
    sha256sum "$FROZEN"
    ;;
  eval-test)
    [[ -f "$FROZEN" ]] || { printf 'Run freeze after validation decisions are final.\n' >&2; exit 3; }
    for seed in "${SEEDS[@]}"; do
      eval_one "$seed" mixed test
      eval_one "$seed" awgn test
      eval_one "$seed" fso test
      for ablation in common_only common_only_shuffled common_only_random; do
        bash run_v10.sh eval "$ROOT/train_mixed_seed${seed}/best.pth" \
          "$ROOT/eval_test_${ablation}_seed${seed}" "${EVAL_TEST[@]}" \
          --seed "$seed" --ablation "$ablation"
      done
    done
    ;;
  aggregate)
    bash run_v10.sh aggregate "$ROOT/aggregate_test_mixed_vs_awgn.json" \
      --baseline "$ROOT/eval_test_awgn_seed42" "$ROOT/eval_test_awgn_seed123" "$ROOT/eval_test_awgn_seed2026" \
      --candidate "$ROOT/eval_test_mixed_seed42" "$ROOT/eval_test_mixed_seed123" "$ROOT/eval_test_mixed_seed2026"
    bash run_v10.sh aggregate "$ROOT/aggregate_test_mixed_vs_fso.json" \
      --baseline "$ROOT/eval_test_fso_seed42" "$ROOT/eval_test_fso_seed123" "$ROOT/eval_test_fso_seed2026" \
      --candidate "$ROOT/eval_test_mixed_seed42" "$ROOT/eval_test_mixed_seed123" "$ROOT/eval_test_mixed_seed2026"
    ;;
  *) printf 'Unknown phase: %s\n' "$PHASE" >&2; exit 2;;
esac
