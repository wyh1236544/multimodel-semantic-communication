# V9-B.1: mode-specific and Common-aligned selection

V9-B.1 fixes the three failures observed in the V9-B 100-image evaluation:

- Generation and Hybrid use separate acceptance/hallucination heads and separately calibrated thresholds.
- Hybrid img2img strength is 0.35 instead of 0.45.
- Training batches are balanced over `(mode, accept/reject)` groups.
- The decoded Common semantic is explicitly projected to frozen CLIP space and aligned to an offline image/text teacher.
- The prompt pre-gate remains observable but is not enforced by default.

The source image and source Localized Narrative are used only to construct
training labels and the Common alignment teacher. Online decisions receive only
decoded Common, received caption, regression/candidate images and channel state.

## Important

The old V9-B dataset and verifier checkpoint are incompatible. V9-B.1 must
recollect data and retrain. The script uses new output directories and does not
overwrite the V9-B results.

## Server smoke test

```bash
cd /data1/zw/WYH/multimodel
unzip -o v9b1_mode_specific_common_patch_20260901.zip
python -m py_compile \
  collect_v9b_verifier_data.py train_v9b_common_verifier.py \
  v9b_common_selective_eval.py check_v9b_architecture.py \
  models/v9b_common_verifier.py
python check_v9b_architecture.py
chmod +x run_v9b_common_selective.sh

V9B_COLLECT_SAMPLES=30 V9B_EPOCHS=15 V9B_EVAL_SAMPLES=10 \
  bash run_v9b_common_selective.sh all
```

If 30 samples do not contain both Hybrid classes, increase
`V9B_COLLECT_SAMPLES` to 50 for the smoke test.

## Formal run

```bash
bash run_v9b_common_selective.sh collect
bash run_v9b_common_selective.sh train
bash run_v9b_common_selective.sh eval
```

Defaults collect 1000 COCO-train images per SNR at -5/0/5 dB, train the small
verifier for 100 epochs, and evaluate 100 COCO-val images at five SNR values.

Outputs:

```text
experiment_results/v9b1_verifier_data/v9b_verifier_dataset.pt
V9.0B.1_model_weights/common_verifier/v9b_verifier_best.pth
experiment_results/v9b1_common_selective_eval/summary.json
experiment_results/v9b1_common_selective_eval/samples.csv
```

Check per-mode precision/recall, `selective_semantic_gain`, pixel deltas, and
the difference between full Common and `no_common` results.

