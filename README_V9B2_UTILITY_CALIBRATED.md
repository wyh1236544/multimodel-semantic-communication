# V9-B.2: utility-calibrated, mode-expert selective generation

V9-B.2 fixes the failure modes observed in the 50-row V9-B.1 evaluation.

## What changed

1. Generation and Hybrid now have dedicated hidden experts, not only separate
   final logits.
2. Auxiliary pixel-safety targets use the same mode-specific limits as data
   collection and oracle evaluation.
3. A continuous decision-utility head learns semantic gain together with the
   cost of exceeding the mode's PSNR/SSIM budget.
4. Calibration searches every validation decision boundary and learns how much
   the utility estimate should contribute. It no longer uses a coarse 0.01 grid.
5. Train/validation splitting is grouped by image ID and requires both classes
   in both modes. Small, invalid experiments fail instead of silently producing
   an all-reject checkpoint.
6. Common dropout is now a true in-distribution Common ablation. Full and
   no-Common models receive separately calibrated score weights and thresholds.

Old V9-B.1 verifier checkpoints are not compatible with the new architecture.
The V9-A reconstruction checkpoint remains unchanged.

## Install

From `/data1/zw/WYH/multimodel`:

```bash
unzip -o v9b2_utility_calibrated_patch_20260903.zip
chmod +x run_v9b_common_selective.sh
python -m py_compile models/v9b_common_verifier.py \
  train_v9b_common_verifier.py collect_v9b_verifier_data.py \
  v9b_common_selective_eval.py check_v9b_architecture.py
python check_v9b_architecture.py
```

## Recommended run

Choose a free physical GPU first. After `CUDA_VISIBLE_DEVICES` is set, the
selected physical card appears as `cuda:0` inside Python.

```bash
nvidia-smi
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 V9B_COLLECT_SAMPLES=300 \
  bash run_v9b_common_selective.sh collect

CUDA_VISIBLE_DEVICES=0 V9B_EPOCHS=100 \
  bash run_v9b_common_selective.sh train

CUDA_VISIBLE_DEVICES=0 V9B_EVAL_SAMPLES=200 \
  bash run_v9b_common_selective.sh eval
```

Run the stages separately so expensive diffusion data do not need to be
recomputed after a training-only adjustment.

For a pipeline smoke test only:

```bash
CUDA_VISIBLE_DEVICES=0 V9B_COLLECT_SAMPLES=80 \
  bash run_v9b_common_selective.sh collect
CUDA_VISIBLE_DEVICES=0 V9B_ALLOW_SMALL_DATASET=1 V9B_EPOCHS=10 \
  bash run_v9b_common_selective.sh train
CUDA_VISIBLE_DEVICES=0 V9B_EVAL_SAMPLES=20 \
  bash run_v9b_common_selective.sh eval
```

Do not report smoke-test numbers in a paper.

## Outputs

- Dataset: `experiment_results/v9b2_verifier_data/v9b_verifier_dataset.pt`
- Labels: `experiment_results/v9b2_verifier_data/v9b_verifier_labels.csv`
- Best verifier: `V9.0B.2_model_weights/common_verifier/v9b_verifier_best.pth`
- Evaluation summary: `experiment_results/v9b2_common_selective_eval/summary.json`
- Per-image results and panels: `experiment_results/v9b2_common_selective_eval/`

## Success criteria

Treat V9-B.2 as successful only if an independent validation/test run shows:

- Hybrid recall and F1 are both non-zero and Hybrid decision utility exceeds
  the regression-only value of zero.
- Generation preserves positive semantic gain while losing materially less
  PSNR/SSIM than forced generation.
- Full Common outperforms separately calibrated no-Common on at least two of
  balanced accuracy, F1, decision utility and false-positive rate.
- Results remain directionally consistent over at least three random seeds and
  at least 500 unique validation images for the paper experiment.
