# V6.5.1：安全分类头、部署加权与文本择优补丁

## 为什么修改

seed442 的 13 个实际候选证明 V6.5 拒绝机制总体正确，但连续的
`ΔCLIP / ΔPSNR / ΔSSIM` 回归头出现均值回归：真实 CLIP 下降的候选仍被
预测为正增益，真实 PSNR 明显下降也未触发对应回归约束。V6.5.1 保留连续
预测用于分析，同时新增三个更容易学习的直接分类目标：

- `semantic_gain_probability`：真实 `ΔCLIP >= 0.003`；
- `psnr_safe_probability`：真实 `ΔPSNR >= -0.05 dB`；
- `ssim_safe_probability`：真实 `ΔSSIM >= -0.004`。

在线接受规则改用这三个安全概率、联合接受概率、幻觉概率和无参考结构特征。

## 其他修改

1. 路由概率越接近实际部署区域，训练样本权重越高；
2. 接收端比较 probe caption 与受保护重传 caption 的无参考置信度，只使用
   更可靠的一条作为生成提示和最终文本；
3. `verification_latency_ms` 现在包含 CLIP 文本/图像一致性特征提取；
4. 输出三个安全概率、文本选择率和 protected/raw 文本指标；
5. 新权重写入 `V6.5.1_model_weights/`，不会覆盖 V6.5。

现有 `experiment_results/v65_verifier_data/verifier_dataset.pth` 可以直接重用，
不需要重新运行扩散数据采集。

## 上传与检查

```bash
cd /data1/zw/WYH/multimodel
unzip -o V6.5.1_safety_heads_text_selector_patch.zip
chmod +x run_v65_server_pipeline.sh
conda activate torch
bash run_v65_server_pipeline.sh check
```

正常输出应包含：

```text
V6.5.1 architecture smoke test: PASS
outputs=[..., 'semantic_gain_probability', 'psnr_safe_probability', 'ssim_safe_probability']
```

## 直接重训

```bash
V651_VERIFIER_EPOCHS=140 \
V651_DEPLOYMENT_WEIGHT=2.0 \
bash run_v65_server_pipeline.sh train
```

日志：

```bash
tail -n 35 logs/v651/04_verifier_train.log
```

除原指标外，重点看：

- `sem`：语义增益分类准确率；
- `psnr_safe`：PSNR 安全分类准确率；
- `ssim_safe`：SSIM 安全分类准确率。

## 重新校准

```bash
bash run_v65_server_pipeline.sh calibrate
cat V6.5.1_model_weights/verifier/verifier_thresholds.json
```

校准只使用 checkpoint 内记录的 12 张验证图像，不使用 seed442 或后续留出集。

## 全新 seed642 试验

```bash
V651_EVAL_SEED=642 \
V651_EVAL_SAMPLES=12 \
V651_ROUTE_THRESHOLD=0.70 \
V651_TEXT_SAFETY=hybrid \
bash run_v65_server_pipeline.sh evaluate
```

结果：

```text
experiment_results/v651_holdout_seed642/
```

```bash
cat experiment_results/v651_holdout_seed642/summary.csv
```

这次不要在 seed642 上修改阈值。先比较三个阶段：回归、第一阶段生成候选、
V6.5.1 最终输出，并检查文本择优是否减少 BLEU 与 token accuracy 的负增益。
