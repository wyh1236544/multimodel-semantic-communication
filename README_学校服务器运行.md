# V6.1完整带宽实验工程——学校服务器运行教程

## 一、项目用途与已核实配置

本工程补齐了缺失的COCO数据加载、损失、训练、评估、三种子汇总和服务器检查程序。模型代码仍是原V6.1代码，只增加了可调带宽入口。

从用户提供的 `seed42.pth` 中读取到的真实结构是：

- 图像尺寸：128×128；
- 隐藏维度：256；
- 公共token池：16；图像token池：256；文本token池：64；
- 动态私有token预算：272；
- 实际传输token总数：`16+272=288`；
- 原 `latent_dim=16`，所以 `288×16=4608` 个实数信道符号；
- 使用VAE、CLIP文本编码器、Residual Adapter、Reliability Gate和双向跨模态注意力。

请注意：原权重配置引用了已不存在的 `/root/models/...` 和Localized Narratives JSONL。本工程把预训练模型改成Hugging Face模型ID，并使用你已经解压的官方COCO captions JSON。这样可以完成新的公平带宽实验，但不能声称逐样本复现原Localized Narratives结果。

## 二、上传和覆盖代码

把 `V6.1_complete_bandwidth_experiment.zip` 上传到：

```text
/data1/zw/WYH/multimodel/
```

在服务器执行：

```bash
cd /data1/zw/WYH/multimodel
mkdir -p backup_before_experiment
cp -a models backup_before_experiment/models

mkdir -p /tmp/v61_complete
unzip -o V6.1_complete_bandwidth_experiment.zip -d /tmp/v61_complete
cp -a /tmp/v61_complete/. /data1/zw/WYH/multimodel/
```

这不会删除 `COCO2017` 或 `V6.1_model_weights`。

检查目录：

```bash
cd /data1/zw/WYH/multimodel
python tools/check_project.py --root /data1/zw/WYH/multimodel
python check_v61_environment.py
```

正确数据数量应为训练图像118287张、验证图像5000张，权重应有seed42/43/44。

## 三、Python环境

先激活原来的GPU环境，再运行：

```bash
nvidia-smi
python check_v61_environment.py
```

如果只有普通依赖缺失，执行：

```bash
pip install -r requirements_v61_experiment.txt
```

不要在已有CUDA版PyTorch可用时重新安装CPU版torch。如果 `torch.cuda.is_available()` 为False，应先由学校服务器管理员或按照服务器CUDA版本安装匹配的PyTorch。

建议把Hugging Face缓存放到数据盘：

```bash
export HF_HOME=/data1/zw/WYH/huggingface_cache
mkdir -p "$HF_HOME"
```

配置默认使用：

```text
stabilityai/sd-vae-ft-mse
openai/clip-vit-base-patch32
```

第一次建模时会联网下载。如果计算节点不能访问Hugging Face，需要事先下载到本地目录，然后在后续命令增加：

```bash
--vae-path /本地/sd-vae-ft-mse \
--text-model-path /本地/clip-vit-base-patch32
```

## 四、第一步：严格评估原始4608权重

先只评估seed42、32个样本，确认整个程序可运行：

```bash
cd /data1/zw/WYH/multimodel

python test.py \
  --config configs/v61_bandwidth.yaml \
  --checkpoint V6.1_model_weights/seed42.pth \
  --preset original_4608 \
  --output experiment_results/original_4608_smoke.json \
  --snr-list 20 \
  --samples 32 \
  --seed 42 \
  --data-root /data1/zw/WYH/multimodel/COCO2017
```

日志必须显示：

```text
channel_uses=4608
```

而且严格权重加载不能出现missing/skipped。如果显存不足，增加：

```bash
--batch-size 2
```

小评估通过后，运行三个权重、500样本、五个SNR：

```bash
python v61_gate_eval.py \
  --config configs/v61_bandwidth.yaml \
  --checkpoint-dir V6.1_model_weights \
  --preset original_4608 \
  --output-dir experiment_results/original_4608 \
  --samples 500 \
  --seeds 42 43 44 \
  --snr-list 0 5 10 15 20 \
  --data-root /data1/zw/WYH/multimodel/COCO2017
```

## 五、第二步：6912带宽冒烟训练

6912把每个token的符号维度从16增加到24。程序会加载原权重的全部相同形状参数，并对变宽的投影层复制重叠部分；新增维度保留随机初始化。扩展详情保存在 `load_report_seed42.json`。

```bash
python v61_gate_train.py \
  --config configs/v61_bandwidth.yaml \
  --preset high_6912 \
  --checkpoint V6.1_model_weights \
  --output-dir experiment_results/high_6912_smoke \
  --seeds 42 \
  --smoke-test \
  --data-root /data1/zw/WYH/multimodel/COCO2017
```

必须看到：

```text
channel_uses: 6912
expanded=若干项
```

并且最终生成：

```text
experiment_results/high_6912_smoke/seed42.pth
```

## 六、第三步：正式训练6912

先只正式训练seed42以筛选带宽：

```bash
python v61_gate_train.py \
  --config configs/v61_bandwidth.yaml \
  --preset high_6912 \
  --checkpoint V6.1_model_weights \
  --output-dir experiment_results/high_6912 \
  --train-samples 10000 \
  --val-samples 500 \
  --epochs 8 \
  --seeds 42 \
  --data-root /data1/zw/WYH/multimodel/COCO2017
```

训练完成后评估：

```bash
python test.py \
  --config configs/v61_bandwidth.yaml \
  --checkpoint experiment_results/high_6912/seed42.pth \
  --preset high_6912 \
  --output experiment_results/high_6912/seed42_eval.json \
  --snr-list 0 5 10 15 20 \
  --samples 500 \
  --seed 42 \
  --data-root /data1/zw/WYH/multimodel/COCO2017
```

如果20 dB PSNR相对4608明显上升，再补三个种子：

```bash
python v61_gate_train.py \
  --config configs/v61_bandwidth.yaml \
  --preset high_6912 \
  --checkpoint V6.1_model_weights \
  --output-dir experiment_results/high_6912 \
  --train-samples 10000 \
  --val-samples 500 \
  --epochs 8 \
  --seeds 42 43 44 \
  --data-root /data1/zw/WYH/multimodel/COCO2017
```

注意：再次包含seed42会重新训练并覆盖同名最佳权重。若seed42已经完成，只运行 `--seeds 43 44`。

三个6912权重齐全后，生成三种子统一评估：

```bash
python v61_gate_eval.py \
  --config configs/v61_bandwidth.yaml \
  --checkpoint-dir experiment_results/high_6912 \
  --preset high_6912 \
  --output-dir experiment_results/high_6912_eval \
  --samples 500 \
  --seeds 42 43 44 \
  --snr-list 0 5 10 15 20 \
  --data-root /data1/zw/WYH/multimodel/COCO2017
```

## 七、继续训练8064和9216

把命令中的preset和输出目录分别替换为：

```text
very_high_8064  → experiment_results/very_high_8064
ultra_9216      → experiment_results/ultra_9216
```

建议顺序为：4608基线 → 6912 seed42 → 8064 seed42 → 必要时9216 seed42。先选出合理带宽，再补三种子，避免浪费GPU时间。

每个档位完成三个权重后，都用与6912相同的 `v61_gate_eval.py` 命令评估，只替换 `--checkpoint-dir`、`--preset` 和 `--output-dir`。

## 八、消融评估

正式模型可以同时评估五种语义流设置：

```bash
python test.py \
  --config configs/v61_bandwidth.yaml \
  --checkpoint experiment_results/high_6912/seed42.pth \
  --preset high_6912 \
  --output experiment_results/high_6912/seed42_ablations.json \
  --snr-list 10 20 \
  --samples 500 \
  --ablations full no_common common_only no_image_private no_text_private \
  --data-root /data1/zw/WYH/multimodel/COCO2017
```

## 九、汇总带宽实验

所有三种子评估完成后：

```bash
python analysis/summarize_bandwidth.py \
  --inputs \
    original_4608=experiment_results/original_4608/summary.json \
    high_6912=experiment_results/high_6912_eval/summary.json \
    very_high_8064=experiment_results/very_high_8064_eval/summary.json \
  --output experiment_results/bandwidth_summary.csv
```

最终比较PSNR、SSIM、token accuracy和BLEU-2，并始终同时报告 `channel_uses`。6912或9216与DeepJSCC的4608不是等带宽比较，只能称为“增加带宽后PSNR向DeepJSCC靠近”。

## 十、常见错误

### CUDA显存不足

在命令后加：

```bash
--batch-size 2
```

仍不足则用1；配置已使用梯度累积。

### 找不到预训练模型

说明服务器无法访问Hugging Face或模型路径不存在。把VAE和CLIP模型上传到服务器，并使用 `--vae-path`、`--text-model-path`。

### 权重尺寸不匹配

评估时preset必须和权重一致。三个现有权重只能用 `original_4608`；高带宽训练脚本会显式进行shape expansion，高带宽评估则必须加载相同高带宽训练产生的权重。

### PSNR与旧报告不完全一致

当前数据使用官方COCO captions，而原权重配置使用未提供的Localized Narratives JSONL；图像子集、描述选择和随机VAE采样也会影响结果。所有新旧带宽必须使用本工程相同的500条验证样本，才能进行公平内部比较。
