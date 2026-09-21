# V6.4：无参考自适应 ControlNet 恢复

## V6.3 证据

12 个 oracle pilot 样本中，ControlNet Tile + 保守融合的生成正样本比例为
66.7%。整体 CLIP image 增加约 0.0288，而 PSNR 只变化约 -0.0107 dB。
这证明空间约束生成专家具有潜力，但 V6.3 使用原图选择候选，只能作为
可行性证据，不能作为最终接收机性能。

## V6.4 的部署逻辑

V6.4 把 oracle 决策蒸馏到只读取接收端特征的轻量策略。策略包含五个动作：

1. 固定总 token 预算内的文字比例；
2. 回归或 ControlNet 生成路由；
3. 扩散强度；
4. RGB 融合或 detail 融合；
5. 保守融合系数 alpha。

数据收集阶段使用原图生成标签。最终评估严格排除所有策略数据 image ID，
并且推理策略不读取原图、源端 mask 或参考质量指标。

## 文件

- `models/controlnet_fusion_policy.py`
- `collect_v64_controlnet_policy_data.py`
- `train_v64_controlnet_policy.py`
- `v64_controlnet_adaptive_eval.py`
- `check_v64_policy_architecture.py`
- `run_v64_server_pipeline.sh`

依赖 V6.3 的：

- `v63_controlnet_fusion_pilot.py`
- `check_v63_controlnet_environment.py`

## 上传和检查

解压到 `/data1/zw/WYH/multimodel/`，保持 `models/` 目录：

```bash
cd /data1/zw/WYH/multimodel
unzip -o V6.4_controlnet_receiver_policy.zip
conda activate torch
bash run_v64_server_pipeline.sh check
```

## 小规模完整闭环

不要直接运行 `quick`。按阶段运行并检查每一步：

```bash
V64_POLICY_SAMPLES=20 bash run_v64_server_pipeline.sh collect
bash run_v64_server_pipeline.sh policy
V64_EVAL_SAMPLES=12 bash run_v64_server_pipeline.sh evaluate
```

收集阶段默认 SNR 为 `-5/0/5/10/15/20 dB`，每个 SNR 使用相同的 20 个
训练图像，共生成 120 条策略记录。每条记录扫描三个扩散强度、两种融合专家
和三个 alpha。只有三个 ControlNet 前向；融合候选是廉价的张量运算。

## 关键结果

策略训练日志至少需要同时报告：

- routing balanced accuracy 和 F1；
- text utility MAE；
- positive 样本的 strength/alpha MAE；
- positive 样本的 fusion expert accuracy。

最终 `experiment_results/v64_holdout/summary.csv` 重点检查：

- `policy_data_overlap_count` 必须为 0；
- `routing_clip_gain` 是否为正；
- `routing_psnr_gain`、`routing_ssim_gain` 是否接近 0 或为正；
- 低 SNR generation rate 是否高于高 SNR，但不能简单等同于 SNR 阈值；
- 文字准确率是否因重复保护而提高。

如果 20 图 pilot 的验证集只有一个路由类别，训练脚本会主动拒绝训练；这时应
把 `V64_POLICY_SAMPLES` 增加到 50，而不是绕过检查。
