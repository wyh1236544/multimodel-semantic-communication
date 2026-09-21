# Localized Narratives migration

All new training and evaluation runs use the COCO Localized Narratives files:

```text
/data1/zw/WYH/multimodel/COCO2017/annotations/localized_narratives/
├── coco_train_captions.jsonl
└── coco_val_captions.jsonl
```

The original COCO image directories remain unchanged:

```text
/data1/zw/WYH/multimodel/COCO2017/train2017
/data1/zw/WYH/multimodel/COCO2017/val2017
```

The default configuration uses the existing offline model directories:

```text
/data1/zw/WYH/multimodel/pretrained/sd-vae-ft-mse
/data1/zw/WYH/multimodel/pretrained/clip-vit-base-patch32
```

This prevents training jobs from attempting to contact Hugging Face.

## What changed

- `data/coco_captions.py` now accepts official COCO `.json` and Localized
  Narratives `.jsonl` annotations.
- Narratives are grouped by `image_id`; multiple narratives for one image stay
  in the same split.
- Training samples one available narrative at random. Validation uses a
  deterministic first narrative.
- `utils/config.py --data-root` now resolves to Localized Narratives, so it can
  no longer silently switch a run back to short COCO Captions.
- `configs/v61_bandwidth.yaml` points to the new annotations.
- The text sequence length remains 64 for compatibility with existing V6.1
  checkpoints. Use the checker to measure truncation before changing the model
  to 77-token or chunked text encoding.

## Validate the data before training

Run the structural and image-ID check:

```bash
cd /data1/zw/WYH/multimodel

python tools/check_localized_narratives.py \
  --data-root /data1/zw/WYH/multimodel/COCO2017 \
  --output experiment_results/localized_narratives_data_report.json
```

Expected annotation counts:

```text
train: 134272
val:     8573
```

For exact CLIP token-length and truncation statistics, use the same local CLIP
model as the communication system:

```bash
python tools/check_localized_narratives.py \
  --data-root /data1/zw/WYH/multimodel/COCO2017 \
  --tokenizer /data1/zw/WYH/multimodel/pretrained/clip-vit-base-patch32 \
  --max-length 64 \
  --skip-image-check \
  --output experiment_results/localized_narratives_clip_length_report.json
```

If that pretrained path differs, pass the actual path used by
`model.text_model_name`.

## Smoke-test the new baseline

The first run should remain deterministic regression only. Do not invoke the
diffusion, ControlNet, verifier, or structure-side-channel pipelines yet.

```bash
python train.py \
  --config configs/v61_bandwidth.yaml \
  --preset original_4608 \
  --output-dir experiment_results/ln_v61_smoke \
  --init-checkpoint V6.1_model_weights/seed42.pth \
  --seeds 42 \
  --smoke-test \
  --data-root /data1/zw/WYH/multimodel/COCO2017
```

Then evaluate the smoke checkpoint:

```bash
python test.py \
  --config configs/v61_bandwidth.yaml \
  --checkpoint experiment_results/ln_v61_smoke/seed42.pth \
  --preset original_4608 \
  --output experiment_results/ln_v61_smoke/eval.json \
  --snr-list 0 10 20 \
  --ablations full no_common common_only no_image_private no_text_private \
  --samples 64 \
  --seed 42 \
  --data-root /data1/zw/WYH/multimodel/COCO2017
```

Only after the data check and smoke run pass should a full Localized Narratives
training run be started.
