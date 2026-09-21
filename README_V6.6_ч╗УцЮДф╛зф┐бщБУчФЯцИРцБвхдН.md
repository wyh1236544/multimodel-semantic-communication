# V6.6：低码率结构侧信道辅助的生成式语义恢复

## 研究问题

V6.5.1 在 seed642 上将 7 个生成候选全部拒绝。验证器避免了结构损失，
但同时丢掉了大象等有价值的语义恢复。核心原因不是单纯阈值不合适，而是
扩散模型只看到严重损坏的回归图，缺少可信的发送端空间布局。

V6.6 将 img2img 初始图和 ControlNet 条件分离：

- img2img 初始图：接收端 V6.1 回归图；
- 文本条件：保留 V6.5.1 的 probe/protected 无参考择优；
- ControlNet 条件：发送端原图提取的低分辨率二值结构图；
- 原图像素：只用于离线 PSNR、SSIM、CLIP 和 oracle 选择，绝不作为生成输入。

结构图使用不依赖 OpenCV 的 Sobel 提取器，默认大小为 `64x64`，即 4096 个
源比特。`oracle` 表示这 4096 bit 无误码送达，是正式结构信道实验前的上界；
`noisy` 使用重复 BPSK + AWGN，报告实际 BER、重复次数、总信道bit和 bpp。

## 为什么先做 oracle gate

如果无误码发送端结构相对“从接收端回归图重新提边缘”都没有明显优势，继续
设计结构编码或训练策略没有意义。因此实验严格分为两步：

1. `receiver` 与 `oracle` 配对比较；
2. 只有 oracle 通过后才运行 `noisy` 结构信道。

seed642 已被观察，脚本默认使用新的 development seed842，并自动排除：

- V6.4/V6.5.1 训练图像ID；
- `experiment_results/v651_holdout_seed642/results.csv` 中出现过的图像ID。

## 新增文件

```text
utils/v66_structure.py
v66_structure_sideinfo_pilot.py
check_v66_structure_environment.py
inspect_v66_results.py
run_v66_server_pipeline.sh
README_V6.6_结构侧信道生成恢复.md
```

不会覆盖 V6.1—V6.5.1 权重与结果。

## 额外离线模型

需要 Canny ControlNet：

```text
lllyasviel/control_v11p_sd15_canny
```

上传后的服务器目录应为：

```text
/data1/zw/WYH/multimodel/pretrained/control_v11p_sd15_canny/
├── config.json
└── diffusion_pytorch_model.bin  # 或 .safetensors
```

## 上传补丁

在服务器执行：

```bash
cd /data1/zw/WYH/multimodel
unzip -o V6.6_structure_side_channel_patch.zip
chmod +x run_v66_server_pipeline.sh
conda activate torch
```

## 1. 环境检查

```bash
bash run_v66_server_pipeline.sh check
```

正常输出应包含：

```text
structure_side_channel=PASS
controlnet_img2img_api=PASS
offline_canny_controlnet_layout=PASS
offline_canny_controlnet_load=PASS cross_attention_dim=768
V6.6 structure environment: PASS
```

## 2. 第一阶段：无误码结构上界

先只跑6张图、3个SNR和两个扩散强度：

```bash
GPU_ID=0 \
V66_ORACLE_SAMPLES=6 \
bash run_v66_server_pipeline.sh oracle
```

结果目录：

```text
experiment_results/v66_structure_oracle_seed842/
├── summary.csv
├── results.csv
├── paired_comparison.csv
├── manifest.json
└── id*_snr*.png
```

运行自动诊断：

```bash
bash run_v66_server_pipeline.sh inspect
```

输出有三种结论：

- `GO_TO_NOISY_STRUCTURE_CHANNEL`：进入真实结构信道；
- `PROMISING_REVIEW_PANELS_BEFORE_CHANNEL`：先人工检查面板；
- `STOP_OR_REDESIGN_STRUCTURE_CONTROL`：暂停结构信道，重新设计条件方式。

自动结论只是 development gate，不构成论文统计显著性结论。

### 面板含义

```text
Original (metrics only)
Regression
Tx structure
Decoded structure
RX-edge candidate
TX clean candidate
TX noisy candidate（channel阶段才出现）
```

候选标题后的 `+/-` 表示该候选是否超过回归效用加 margin；它不是在线接收器
决策，因为第一阶段仍属于 oracle 可行性试验。

## 3. 第二阶段：结构图过AWGN信道

只有第一阶段通过后运行：

```bash
GPU_ID=0 \
V66_CHANNEL_SAMPLES=12 \
bash run_v66_server_pipeline.sh channel
```

默认重复保护：

```text
SNR <= -5 dB: repetition 7
SNR <=  0 dB: repetition 5
SNR <=  5 dB: repetition 3
SNR >   5 dB: repetition 1
```

若要固定重复次数，例如5次：

```bash
V66_STRUCTURE_REPETITION=5 bash run_v66_server_pipeline.sh channel
```

## 关键指标

优先比较 `summary.csv` 中同一SNR下的 `receiver`、`oracle`、`noisy`：

- `candidate_clip_image` / `clip_gain`：语义恢复；
- `candidate_source_structure_f1`：候选与发送端结构的一致性；
- `metric_safe_positive_rate`：同时满足 CLIP、PSNR、SSIM 安全预算的比例；
- `structure_ber`：结构信道误码率；
- `mean_structure_channel_bits` / `mean_structure_bpp`：额外带宽；
- `mean_generation_latency_ms`：所有强度网格的总生成开销。

默认安全预算与 V6.5.1 标签一致：

```text
ΔCLIP >= +0.003
ΔPSNR >= -0.05 dB
ΔSSIM >= -0.004
```

## 第一阶段通过后的正式路线

1. 比较 `32x32` 与 `64x64` 结构图；
2. 比较固定重复和SNR自适应重复；
3. 将结构一致性特征加入新的路由/验证数据集；
4. 按 image ID 分组扩充到至少200张训练图 × 6个SNR；
5. 最终使用从未观察的新 seed1042 做三随机种子留出评估。

不要在 seed842 上反复修改阈值后将其作为最终测试结果。
