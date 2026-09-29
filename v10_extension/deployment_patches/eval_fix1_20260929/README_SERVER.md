# V10 评估错配修复：服务器安装

本包只更新 V10 评估代码、正式实验入口、测试和说明；不改训练权重、划分、旧结果或 V9 文件。包内安装器先检查所有目标的旧版本哈希，遇到不同版本会在写入前停止。旧文件备份在 `v10_extension/deployment_records/eval_fix1_*/backup/`。

先把 `multimodel_v10_eval_fix1_20260929.tar.gz` 和同名 `.sha256` 上传到现有 `/data1/zw/WYH/multimodel` 目录，再执行：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c multimodel_v10_eval_fix1_20260929.tar.gz.sha256
test ! -e v10_extension/deployment_patches/eval_fix1_20260929
tar -xzf multimodel_v10_eval_fix1_20260929.tar.gz
"$PYTHON_BIN" -B v10_extension/deployment_patches/eval_fix1_20260929/install.py --project-root . --dry-run
"$PYTHON_BIN" -B v10_extension/deployment_patches/eval_fix1_20260929/install.py --project-root .
bash run_v10.sh check
```

任何一步报错立即停止并保留现场，不删旧目录、不手工覆盖。特别是哈希冲突说明服务器代码不同，须先核对。

CPU 检查通过后，选物理 GPU1，只做 4 张图的小规模验证：

```bash
export CUDA_VISIBLE_DEVICES=1
export FORMAL_TAG=paper_v10_02_noamp
export V10_ROOT="experiment_results/v10_formal_${FORMAL_TAG}"
bash run_v10.sh eval "$V10_ROOT/train_mixed_seed42/best.pth" \
  "$V10_ROOT/smoke_shuffled_fix1_seed42" \
  --split validation --samples 4 --profiles awgn fso_moderate \
  --snr-list 0 10 --repeats 1 --save-images 2 --no-amp \
  --seed 42 --ablation common_only_shuffled
"$PYTHON_BIN" -B v10_extension/verify_shuffled_v10.py \
  "$V10_ROOT/smoke_shuffled_fix1_seed42"
```

小规模验收通过后，补跑三个种子的正式错配 validation：

```bash
bash run_v10_formal_suite.sh eval-validation-shuffled-fix1
```

每个新目录有 500 张图 × 25 条件 × 5 次重复 = 62,500 行。完成后先交回三个新目录的 `summary.json`、`run_identity.json`、`COMPLETE.json` 及 `samples.jsonl`；**不要立即执行 `freeze` 或 `eval-test`**。只有验证结果核查并书面确定论文主张与展示规则后才冻结。旧的三个 `eval_validation_common_only_shuffled_seed*` 目录必须保留，但其结果无效。
