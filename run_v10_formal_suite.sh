#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PHASE="${1:-help}"
TAG="${FORMAL_TAG:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/zw/anaconda3/envs/torch/bin/python}"
SEEDS=(42 123 2026)
# Full precision is intentional: the server AMP path produced a nonfinite
# gradient during the first formal mixed run, while the bounded no-AMP pilot
# completed. Keep training and evaluation numerics consistent.
TRAIN=(--epochs 6 --train-samples 10000 --val-samples 500 --batch-size 2 --grad-accum 2 --num-workers 0 --max-hours 12 --no-amp)
EVAL_VAL=(--split validation --samples 500 --repeats 5 --save-images 12 --no-amp)
EVAL_TEST=(--split test --samples 1000 --repeats 5 --save-images 20 --no-amp)

if [[ "$PHASE" == help || -z "$TAG" ]]; then
  printf '%s\n' \
    'Usage: FORMAL_TAG=paper_v10_01 CUDA_VISIBLE_DEVICES=1 bash run_v10_formal_suite.sh PHASE' \
    'Phases: preflight, train-core, train-ablation, eval-validation, eval-validation-shuffled-fix1, freeze, eval-test, aggregate' \
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
eval_shuffled_one() {
  local seed="$1" split="$2"
  local budget=("${EVAL_VAL[@]}"); [[ "$split" == test ]] && budget=("${EVAL_TEST[@]}")
  local suffix=common_only_shuffled_fix1
  [[ "$split" == test ]] && suffix=common_only_shuffled
  bash run_v10.sh eval "$ROOT/train_mixed_seed${seed}/best.pth" \
    "$ROOT/eval_${split}_${suffix}_seed${seed}" "${budget[@]}" \
    --seed "$seed" --ablation common_only_shuffled
}

case "$PHASE" in
  preflight)
    bash run_v10.sh check
    if [[ -f v10_extension/manifests/ln_v10_splits.json ]]; then
      printf 'Reusing immutable split manifest: %s\n' \
        'v10_extension/manifests/ln_v10_splits.json'
      sha256sum v10_extension/manifests/ln_v10_splits.json
    else
      bash run_v10.sh splits
    fi
    bash run_v10.sh smoke "$ROOT/smoke_seed42" --batch-size 2 --num-workers 0 --seed 42 --no-amp
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
      for ablation in common_only common_only_random; do
        bash run_v10.sh eval "$ROOT/train_mixed_seed${seed}/best.pth" \
          "$ROOT/eval_validation_${ablation}_seed${seed}" "${EVAL_VAL[@]}" \
          --seed "$seed" --ablation "$ablation"
      done
      eval_shuffled_one "$seed" validation
    done
    ;;
  eval-validation-shuffled-fix1)
    for seed in "${SEEDS[@]}"; do
      eval_shuffled_one "$seed" validation
    done
    ;;
  freeze)
    [[ ! -e "$FROZEN" ]] || { printf 'Freeze record already exists: %s\n' "$FROZEN" >&2; exit 3; }
    for seed in "${SEEDS[@]}"; do
      for name in mixed awgn fso no_uep no_adapter no_common_probe common_only common_only_random common_only_shuffled_fix1; do
        local_dir="$ROOT/eval_validation_${name}_seed${seed}"
        [[ -f "$local_dir/COMPLETE.json" ]] || { printf 'Missing validation: %s\n' "$local_dir" >&2; exit 3; }
        bash run_v10.sh verify "$local_dir"
      done
      "$PYTHON_BIN" -B v10_extension/verify_shuffled_v10.py \
        "$ROOT/eval_validation_common_only_shuffled_fix1_seed${seed}"
    done
    printf '{"tag":"%s","seeds":[42,123,2026],"validation_shuffled_control":"common_only_shuffled_fix1","test_arms":["mixed","awgn","fso","no_uep","no_adapter","no_common_probe","common_only","common_only_random","common_only_shuffled"],"test_policy":"single frozen evaluation; no tuning after test","status":"frozen"}\n' "$TAG" > "$FROZEN"
    sha256sum "$FROZEN"
    ;;
  eval-test)
    [[ -f "$FROZEN" ]] || { printf 'Run freeze after validation decisions are final.\n' >&2; exit 3; }
    for seed in "${SEEDS[@]}"; do
      eval_one "$seed" mixed test
      eval_one "$seed" awgn test
      eval_one "$seed" fso test
      eval_one "$seed" no_uep test --no-uep
      eval_one "$seed" no_adapter test --no-adapter
      eval_one "$seed" no_common_probe test
      for ablation in common_only common_only_random; do
        bash run_v10.sh eval "$ROOT/train_mixed_seed${seed}/best.pth" \
          "$ROOT/eval_test_${ablation}_seed${seed}" "${EVAL_TEST[@]}" \
          --seed "$seed" --ablation "$ablation"
      done
      eval_shuffled_one "$seed" test
    done
    ;;
  aggregate)
    bash run_v10.sh aggregate "$ROOT/aggregate_test_mixed_vs_awgn.json" \
      --baseline "$ROOT/eval_test_awgn_seed42" "$ROOT/eval_test_awgn_seed123" "$ROOT/eval_test_awgn_seed2026" \
      --candidate "$ROOT/eval_test_mixed_seed42" "$ROOT/eval_test_mixed_seed123" "$ROOT/eval_test_mixed_seed2026"
    bash run_v10.sh aggregate "$ROOT/aggregate_test_mixed_vs_fso.json" \
      --baseline "$ROOT/eval_test_fso_seed42" "$ROOT/eval_test_fso_seed123" "$ROOT/eval_test_fso_seed2026" \
      --candidate "$ROOT/eval_test_mixed_seed42" "$ROOT/eval_test_mixed_seed123" "$ROOT/eval_test_mixed_seed2026"
    for name in no_uep no_adapter no_common_probe; do
      bash run_v10.sh aggregate "$ROOT/aggregate_test_mixed_vs_${name}.json" \
        --baseline "$ROOT/eval_test_mixed_seed42" "$ROOT/eval_test_mixed_seed123" "$ROOT/eval_test_mixed_seed2026" \
        --candidate "$ROOT/eval_test_${name}_seed42" "$ROOT/eval_test_${name}_seed123" "$ROOT/eval_test_${name}_seed2026"
    done
    for name in common_only_random common_only_shuffled; do
      bash run_v10.sh aggregate "$ROOT/aggregate_test_common_only_vs_${name}.json" \
        --baseline "$ROOT/eval_test_${name}_seed42" "$ROOT/eval_test_${name}_seed123" "$ROOT/eval_test_${name}_seed2026" \
        --candidate "$ROOT/eval_test_common_only_seed42" "$ROOT/eval_test_common_only_seed123" "$ROOT/eval_test_common_only_seed2026"
    done
    ;;
  *) printf 'Unknown phase: %s\n' "$PHASE" >&2; exit 2;;
esac
