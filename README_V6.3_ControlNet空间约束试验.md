# V6.3：ControlNet Tile 空间约束生成恢复试验

## 为什么升级

V6.2 的真实 pilot 表明，prompt-only 和全局语义 token 只能轻微改变语义，
无法纠正 0 dB 回归图中的人物、家具和背景结构错位。仅做 RGB 加权或高频
融合也不能解决低频结构错误。因此 V6.3 先验证一个新的生成专家：

1. 接收文字作为 Stable Diffusion 的语义条件；
2. 接收端回归图作为 img2img 初始图；
3. 同一回归图同时作为 ControlNet Tile 的空间条件；
4. 扫描扩散强度，并比较 raw、RGB 融合和仅细节融合；
5. 原图只用于 pilot 的 oracle 评分，不作为接收端输入。

这一步回答的是“新的生成专家是否具备正收益潜力”。它不是最终部署策略，
因此不能把 oracle 指标直接写成最终系统性能。

## 新增离线模型

下载 Hugging Face 仓库：

`lllyasviel/control_v11f1e_sd15_tile`

服务器目标目录：

`/data1/zw/WYH/multimodel/pretrained/control_v11f1e_sd15_tile`

至少需要：

- `config.json`
- `diffusion_pytorch_model.bin` 或 `diffusion_pytorch_model.safetensors`

## 上传代码

把压缩包内容上传到：

`/data1/zw/WYH/multimodel/`

新增的三个可执行文件都位于项目根目录，不覆盖 V6.2 checkpoint。

## 检查

```bash
cd /data1/zw/WYH/multimodel
conda activate torch
bash run_v63_controlnet_pilot.sh check
```

预期最后显示：

```text
controlnet_img2img_api=PASS
offline_controlnet_layout=PASS
offline_controlnet_load=PASS cross_attention_dim=768
V6.3 ControlNet environment: PASS
```

## Pilot

```bash
bash run_v63_controlnet_pilot.sh pilot
```

默认只运行 6 张图、SNR 0/10 dB。每个样本扫描扩散强度
`0.25/0.45/0.65`，并扫描融合系数 `0.25/0.50/0.75`。

完成后：

```bash
cat experiment_results/v63_controlnet_pilot/summary.csv
```

重点看：

- `oracle_generation_positive_rate`
- `oracle_utility_gain`
- `oracle_psnr_gain`
- `oracle_ssim_gain`
- `oracle_clip_gain`

只有当 0 dB 下存在稳定的正样本比例，并且视觉结果不是无意义幻觉时，
才把 ControlNet/Fusion 参数加入最终接收端策略训练。否则应停止扩散路线，
转向增强回归解码器本身。
