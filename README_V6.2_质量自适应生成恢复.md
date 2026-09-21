# V6.2：质量自适应带宽与生成恢复

## 1. 这次为什么不再只按 SNR 切换

旧实验把 `SNR <= threshold` 直接等同于“使用扩散”。这会产生两个问题：

1. 同一 SNR 下，不同图像的恢复难度差别很大；
2. 如果接收文本本身不可靠，扩散模型会被错误 prompt 引导，生成清晰但不真实的内容。

V6.2 改为一个闭环策略，并将额外文本带宽实现为有效 token 的重复保护，而不是发送 padding：

```text
V6.1 初次恢复（中性图文配额）
  -> 16 维无参考质量/可靠性指标
  -> 文本不清楚：提高 text utility，固定总 token 数内增加文本 token，再恢复一次
  -> 再次评估
  -> 图像质量差且文本可信：语义条件扩散
  -> 否则：回归恢复
```

初次和二次传输使用相同的总模态 token 预算 272。提高文本带宽并不是凭空增加总带宽，而是从图像私有 token 中动态让出一部分给文本。COCO caption 通常少于 16 个有效 token，因此多出的文本信道位置会重复发送高重要度的有效 token，接收端对独立噪声副本求平均，形成文本 unequal-error protection（UEP）。`text_repetition_factor` 会直接报告平均重复保护倍数。代码同时保留 `feedback_used`，因此论文中应准确称为 **receiver-feedback-assisted adaptive allocation**。若以后做视频/连续帧，可使用上一帧反馈实现单次传输；静态图像实验则把第一次传输视为 probe/首轮 HARQ。

## 2. 新增文件

- `models/quality_adaptive_recovery.py`：19 维质量特征、质量路由器、文本带宽策略、扩散语义适配器。
- `models/shared_bottleneck.py`：加入有效文本 token 重复发送与接收端合并。
- `models/joint_image_text_model.py`：输出真实文本冗余度和接收 EOS 信息。
- `utils/v62_diffusion.py`：兼容旧版 diffusers 的本地 SD-v1.5 prompt/语义条件工具。
- `train_v62_semantic_adapter.py`：冻结 V6.1 和 SD，只训练约束扩散的语义 token 适配器。
- `collect_v62_policy_data.py`：穷举文本配额和扩散强度，生成 policy 的 oracle 标签。
- `train_v62_quality_policy.py`：训练最终接收端策略器。
- `v62_quality_adaptive_eval.py`：完整闭环评估和可视化。
- `check_v62_architecture.py`：不加载预训练模型的 CPU 快速自检。
- `check_v62_diffusion_environment.py`：检查离线模型布局和语义 prompt API，防止适配器被静默忽略。
- `data/coco_captions.py`：支持按 image id 排除策略数据，保证最终测试严格独立。
- `plot_v62_results.py`：将严格 holdout 的 `summary.csv` 绘制为组会四联图。
- `run_v62_server_pipeline.sh`：服务器分阶段运行入口，自动保存每阶段日志。

不需要再下载新的基础模型。继续使用：

```text
pretrained/sd-vae-ft-mse
pretrained/clip-vit-base-patch32
pretrained/stable-diffusion-v1-5
V6.1_model_weights/seed42.pth
```

## 3. 上传方法

将压缩包中的 13 个代码文件按相同相对目录上传到（`models` 和 `data` 中的旧文件需要覆盖）：

```text
/data1/zw/WYH/multimodel/
```

例如 `models/quality_adaptive_recovery.py` 必须位于：

```text
/data1/zw/WYH/multimodel/models/quality_adaptive_recovery.py
```

进入服务器项目目录：

```bash
cd /data1/zw/WYH/multimodel
conda activate torch
```

先做语法检查：

```bash
python -m py_compile \
  models/quality_adaptive_recovery.py \
  models/shared_bottleneck.py \
  models/joint_image_text_model.py \
  data/coco_captions.py \
  utils/v62_diffusion.py \
  train_v62_semantic_adapter.py \
  collect_v62_policy_data.py \
  train_v62_quality_policy.py \
  v62_quality_adaptive_eval.py \
  check_v62_architecture.py \
  check_v62_diffusion_environment.py \
  plot_v62_results.py \
  run_v62_server_pipeline.sh
```

然后先运行无需数据和预训练权重的结构自检：

```bash
python check_v62_architecture.py
```

也可以用统一入口逐阶段执行；每一步都会在 `logs/v62/` 保存完整日志：

```bash
bash run_v62_server_pipeline.sh check
bash run_v62_server_pipeline.sh pilot
bash run_v62_server_pipeline.sh adapter
bash run_v62_server_pipeline.sh collect
bash run_v62_server_pipeline.sh policy
bash run_v62_server_pipeline.sh evaluate
```

快速链路默认 adapter 1000 step、每个 SNR 收集 50 张、最终每个 SNR 评估 20 张：

```bash
bash run_v62_server_pipeline.sh quick
```

扩大正式实验时使用环境变量，不需要修改脚本：

```bash
V62_ADAPTER_STEPS=0 V62_ADAPTER_SAMPLES=10000 bash run_v62_server_pipeline.sh adapter
V62_POLICY_SAMPLES=500 bash run_v62_server_pipeline.sh collect
V62_EVAL_SAMPLES=500 bash run_v62_server_pipeline.sh evaluate
```

正常结果必须包含 `V6.2 architecture smoke test: PASS`、`feature_dim=19`、两个不同的文本重复倍数，以及 `0dB_text_mse`。本机 PyTorch 1.12 的实际自检中，文本 utility 从 0 增加到 1 后，0 dB 文本特征 MSE 从约 0.505 降至约 0.125，同时图像 token 从 256 降至 208；这正是策略器需要学习权衡的 text-reliability/image-detail trade-off。

随后检查服务器的离线扩散环境：

```bash
python check_v62_diffusion_environment.py --load-pipeline
```

它必须输出 `semantic_prompt_api=PASS`、`offline_model_layout=PASS`、`offline_pipeline_load=PASS`。如果 `prompt_embeds` API 不存在，不能继续训练后的完整评估；代码会明确报错，而不会把语义适配器悄悄降级成普通 prompt-only SD。工程依赖中规定 `diffusers>=0.20`。

## 4. 建议的实际运行顺序

### A. 先跑无需训练的 12 图小试验

不传 `--policy-checkpoint` 和 `--adapter-checkpoint` 时，程序使用安全初始化策略与 prompt-only 扩散。这一步主要验证闭环文本配额是否工作，不代表最终扩散性能。

```bash
CUDA_VISIBLE_DEVICES=0 python v62_quality_adaptive_eval.py \
  --checkpoint V6.1_model_weights/seed42.pth \
  --data-root COCO2017 \
  --vae-path pretrained/sd-vae-ft-mse \
  --text-model-path pretrained/clip-vit-base-patch32 \
  --diffusion-model-path pretrained/stable-diffusion-v1-5 \
  --clip-model-path pretrained/clip-vit-base-patch32 \
  --snr-list 0 10 20 \
  --samples 12 \
  --batch-size 2 \
  --output-dir experiment_results/v62_pilot
```

重点查看：

```text
experiment_results/v62_pilot/results.csv
experiment_results/v62_pilot/summary.csv
```

确认 `text_utility`、`allocated_text_tokens`、`text_repetition_factor`、`generation_probability` 不再只是 SNR 的固定函数。质量特征还会使用已有 CLIP 计算“恢复图像—接收文本”一致性；它不读取原图，也不需要下载新模型。

### B. 训练接收语义到扩散模型的适配器

先用 1000 step 验证训练链路：

```bash
CUDA_VISIBLE_DEVICES=0 python train_v62_semantic_adapter.py \
  --checkpoint V6.1_model_weights/seed42.pth \
  --data-root COCO2017 \
  --vae-path pretrained/sd-vae-ft-mse \
  --text-model-path pretrained/clip-vit-base-patch32 \
  --diffusion-model-path pretrained/stable-diffusion-v1-5 \
  --train-samples 3000 \
  --max-steps 1000 \
  --batch-size 1 \
  --output-dir V6.2_model_weights/adapter
```

无报错后正式训练建议 `--train-samples 10000 --epochs 3 --max-steps 0`。这一阶段最重要，因为它把 V6.1 接收到的 common/image/text token 变成 8 个 SD cross-attention 条件 token，解决 prompt-only 扩散“重画而非恢复”的问题。

### C. 收集策略监督数据

首次先收集每个 SNR 50 张（总计 300 条），确认流程：

```bash
CUDA_VISIBLE_DEVICES=0 python collect_v62_policy_data.py \
  --checkpoint V6.1_model_weights/seed42.pth \
  --data-root COCO2017 \
  --vae-path pretrained/sd-vae-ft-mse \
  --text-model-path pretrained/clip-vit-base-patch32 \
  --diffusion-model-path pretrained/stable-diffusion-v1-5 \
  --clip-model-path pretrained/clip-vit-base-patch32 \
  --adapter-checkpoint V6.2_model_weights/adapter/semantic_adapter_latest.pth \
  --snr-list -5 0 5 10 15 20 \
  --samples 50 \
  --output-dir experiment_results/v62_policy_data_pilot
```

正式结果至少建议每个 SNR 500 张。脚本会比较文本 utility `[0,.25,.5,.75,1]` 和扩散 strength `[.10,.18,.26,.34]`。监督效用由 SSIM、PSNR、CLIP 图像语义相似度和图文一致性共同构成。

### D. 训练质量/带宽策略器

```bash
CUDA_VISIBLE_DEVICES=0 python train_v62_quality_policy.py \
  --dataset experiment_results/v62_policy_data_pilot/policy_dataset.pth \
  --epochs 80 \
  --output-dir V6.2_model_weights/policy
```

数据按 COCO image id 分层划分训练/验证，避免同一图像在不同 SNR 下跨集合泄漏。训练使用类别平衡的 routing loss，并报告 balanced accuracy、precision、recall 和 F1。如果生成候选从未优于回归，或验证集只有一种 routing 标签，脚本会停止并要求先改善 semantic adapter，而不会训练一个“永远回归”的伪高准确率模型。checkpoint 会保存全部策略数据 image id。

### E. 完整评估

```bash
CUDA_VISIBLE_DEVICES=0 python v62_quality_adaptive_eval.py \
  --checkpoint V6.1_model_weights/seed42.pth \
  --policy-checkpoint V6.2_model_weights/policy/quality_policy_best.pth \
  --adapter-checkpoint V6.2_model_weights/adapter/semantic_adapter_latest.pth \
  --data-root COCO2017 \
  --vae-path pretrained/sd-vae-ft-mse \
  --text-model-path pretrained/clip-vit-base-patch32 \
  --diffusion-model-path pretrained/stable-diffusion-v1-5 \
  --clip-model-path pretrained/clip-vit-base-patch32 \
  --snr-list -5 0 5 10 15 20 \
  --samples 500 \
  --full-ablation \
  --output-dir experiment_results/v62_final
```

完整评估默认读取 policy checkpoint 中保存的 image id，并从 COCO 数据集中排除所有策略采集图像；不要为正式结果添加 `--allow-policy-data-overlap`。`--full-ablation` 会在完全相同的 holdout 图像上同时评价固定回归、SNR 阈值 prompt-only SD、文本自适应回归、质量路由 prompt-only SD 和完整语义适配器方案。prompt-only 与 semantic-adapter 候选复用完全相同的扩散随机噪声，因此条件方式是唯一变量。`summary.csv` 会按 SNR 聚合文本 accuracy/BLEU 增益、反馈前后 PSNR/SSIM、质量路由相对 SNR 阈值的增益、语义适配器相对 prompt-only 的增益、文本重复倍数、反馈率、生成率、实际扩散延迟和 `policy_data_overlap_count`。正式结果要求最后一列始终为 0。

生成组会结果图：

```bash
python plot_v62_results.py \
  --summary experiment_results/v62_final/summary.csv \
  --output experiment_results/v62_final/v62_summary.png
```

绘图只额外依赖 `matplotlib`；若环境没有，执行 `pip install matplotlib`。图标题会明确显示 `STRICT HOLDOUT: PASS` 或数据重叠警告。

## 5. 必须做的对照与消融

主表至少包含：

1. V6.1 regression（固定 utility=0.5）；
2. SNR-threshold + prompt-only SD（旧基线）；
3. quality router + prompt-only SD；
4. quality router + semantic adapter；
5. quality router + semantic adapter + text feedback allocation（完整模型）。

消融建议：去掉文本熵/置信度、去掉跨模态一致性、去掉视觉质量统计、只用 SNR、固定文本 token、不同反馈 deadband。指标必须同时报告 PSNR/SSIM、CLIP image similarity、LPIPS（如服务器已有包）、BLEU/token accuracy、平均文本 token、扩散调用率和平均推理时间。

## 6. 预期与停止条件

不要把“所有低 SNR 图都生成”当成目标。合理结果通常是：

- 生成分支的 PSNR 可能不超过回归分支，但 CLIP/LPIPS 与主观质量改善；
- 文本置信度低时先增加文本 token，生成概率暂时降低；
- 文本稳定且图像结构恢复差时才提高生成概率；
- 高 SNR、结构简单图像即使 SNR 相同也应选择回归。

若训练后生成分支在 CLIP image similarity 与 LPIPS 上仍不能稳定超过回归，先不要继续调路由阈值，应回到语义适配器训练；路由器无法挽救一个本身无优势的生成专家。
