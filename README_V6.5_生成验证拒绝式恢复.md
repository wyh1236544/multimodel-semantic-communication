# V6.5：生成—验证—拒绝式多模态语义通信接收机

## 这一版解决什么问题

V6.4 已经通过校准阈值将生成调用率从约 79% 降到约 21%，但仍有
`id25394` 这类结果：生成图与接收文本在高层语义上相似，却改变了人物、
姿态、物体或场景细节。V6.5 在 V6.4 路由后加入接收端候选验证器：

1. V6.4 首先判断是否值得调用 ControlNet；
2. ControlNet 产生候选并与回归图保守软融合；
3. V6.5 只使用接收端信息评估语义增益和结构漂移；
4. 验证通过才输出生成候选，否则自动退回回归结果。

源图只在数据采集和最终指标统计时使用，不进入在线决策。

## 新增内容

- `models/selective_generation_verifier.py`：预测接受概率、幻觉风险以及
  `ΔCLIP / ΔPSNR / ΔSSIM`。
- `utils/v65_selective.py`：19 项生成后验证特征、混合文本 UEP 和硬安全约束。
- `collect_v65_verifier_data.py`：生成图像 ID 可追踪的监督数据。
- `train_v65_verifier.py`：按图像 ID 隔离训练集和验证集。
- `calibrate_v65_verifier.py`：仅使用验证图像自动校准八项接受约束。
- `v65_generate_verify_eval.py`：严格排除 V6.4/V6.5 所有训练图像的留出评估。
- `inspect_v65_dataset.py`：查看接受/拒绝标签和多指标增益分布。
- `run_v65_server_pipeline.sh`：服务器一键流程。

不需要下载新的预训练模型。继续使用现有 V6.1、SD1.5、ControlNet Tile、
CLIP 和 V6.2 semantic adapter。

## 上传

把压缩包上传到：

```text
/data1/zw/WYH/multimodel/
```

然后执行：

```bash
cd /data1/zw/WYH/multimodel
unzip -o V6.5_generate_verify_reject.zip
chmod +x run_v65_server_pipeline.sh
conda activate torch
```

解压时必须保留 `models/` 和 `utils/` 目录结构。

## 第一步：环境与架构检查

```bash
bash run_v65_server_pipeline.sh check
```

正常时应看到：

```text
V6.5 architecture smoke test: PASS
route_feature_dim=19
verifier_feature_dim=19
V6.3 ControlNet environment: PASS
```

## 第二步：先做 60 张训练图像的试验数据

每张图像在 6 个 SNR 下测试，只有第一阶段概率不低于 0.30 的候选才会
真正调用扩散模型，因此记录数通常小于 `60 × 6`。

```bash
V65_VERIFIER_SAMPLES=60 \
V65_COLLECT_SEED=342 \
bash run_v65_server_pipeline.sh collect
```

采集过程中会持续输出 `accept`、`dCLIP`、`dPSNR` 和 `dSSIM`，并定期写入：

```text
experiment_results/v65_verifier_data/verifier_dataset.partial.pth
```

完成后检查标签：

```bash
bash run_v65_server_pipeline.sh inspect
```

接受和拒绝两类都必须存在。试验阶段如果任一类少于约 15%，先把输出发给我，
不要直接训练；需要根据真实分布调整监督约束。

## 第三步：训练验证器

```bash
V65_VERIFIER_EPOCHS=120 \
bash run_v65_server_pipeline.sh train
```

权重保存在：

```text
V6.5_model_weights/verifier/generation_verifier_best.pth
```

重点关注验证集：`balanced`、`F1`、`FPR`、`hall_acc` 和三项增益 MAE。
本任务优先压低 FPR，因为错误接受一次生成结果比保守回退更危险。

## 第四步：只在验证集上校准约束

```bash
bash run_v65_server_pipeline.sh calibrate
```

输出：

```text
V6.5_model_weights/verifier/verifier_thresholds.json
```

校准同时约束：接受概率、预测语义增益、预测 PSNR/SSIM 风险、幻觉概率、
文本对齐增益、边缘一致性和低频结构漂移。

## 第五步：全新 seed 的严格留出试验

第一次只跑 12 张，确认路由和图片：

```bash
V65_EVAL_SEED=442 \
V65_EVAL_SAMPLES=12 \
V65_ROUTE_THRESHOLD=0.70 \
V65_TEXT_SAFETY=hybrid \
bash run_v65_server_pipeline.sh evaluate
```

结果在：

```text
experiment_results/v65_holdout_seed442/
```

查看汇总：

```bash
cat experiment_results/v65_holdout_seed442/summary.csv
```

列出所有被调用、接受和拒绝的样本：

```bash
python -c "import csv; r=list(csv.DictReader(open('experiment_results/v65_holdout_seed442/results.csv'))); [print(x['snr_db'],x['image_id'],'invoke='+x['primary_invoked'],'accept='+x['verified_accept'],'route='+x['generation_probability'],'verify='+x['acceptance_probability'],x['panel']) for x in r if x['primary_invoked']=='1']"
```

## V6.5 的判断标准

首先检查：

- `training_data_overlap_count=0`；
- `verified_generation_rate <= primary_generation_rate`；
- `verification_rejection_rate > 0`，证明验证器确实在工作；
- `text_accuracy_gain` 总体不低于 0；
- `selected_psnr_gain` 和 `selected_ssim_gain` 比候选阶段更接近 0；
- `selected_clip_gain` 仍为正；
- 被接受的图像没有明显身份、姿态和物体改写。

只有这 12 张试验满足上述条件后，才换另一个从未使用的 seed，扩大到
50–100 张正式留出集，避免在 seed442 上反复调参。

## 正式训练规模

60 张只是验证闭环。结构确认后重新采集 100–200 张训练图像：

```bash
V65_VERIFIER_SAMPLES=150 \
V65_COLLECT_SEED=542 \
V65_COLLECT_OUTPUT=experiment_results/v65_verifier_data_formal \
bash run_v65_server_pipeline.sh collect
```

正式数据应使用新的输出目录和权重目录，不覆盖试验记录。训练时相应指定：

```bash
VERIFIER_DATA=/data1/zw/WYH/multimodel/experiment_results/v65_verifier_data_formal/verifier_dataset.pth \
bash run_v65_server_pipeline.sh train
```

扩大数据后需要重新训练和校准，再在第三组完全独立图像上报告结果。
