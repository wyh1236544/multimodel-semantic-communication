# V9-B Common-guided selective generation

V9-B keeps the trained V9-A semantic-communication backbone and adds a
receiver-side prompt pre-gate plus a Common-guided post-generation verifier.
The source image and Localized Narrative are used only to create offline oracle
labels. They are never verifier inputs.

## Server smoke test

```bash
cd /data1/zw/WYH/multimodel
unzip -o v9b_common_selective_patch_20260901.zip
python -m py_compile v9_feedback_task_adaptive_eval.py \
  collect_v9b_verifier_data.py train_v9b_common_verifier.py \
  v9b_common_selective_eval.py check_v9b_architecture.py \
  models/v9b_common_verifier.py utils/v9b_selective.py
python check_v9b_architecture.py
chmod +x run_v9b_common_selective.sh
V9B_COLLECT_SAMPLES=20 V9B_EPOCHS=10 V9B_EVAL_SAMPLES=10 \
  bash run_v9b_common_selective.sh all
```

## Formal run

```bash
bash run_v9b_common_selective.sh collect
bash run_v9b_common_selective.sh train
bash run_v9b_common_selective.sh eval
```

Defaults collect 500 COCO-train images at -5/0/5 dB and evaluate 100 COCO-val
images at -5/0/5/10/20 dB. The final summary compares forced diffusion,
V9-B selection, an oracle upper bound, and a no-Common ablation.

