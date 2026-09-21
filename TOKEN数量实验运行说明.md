# V6.1 传输 Token 数量控制变量实验

## 实验定义

本实验不改变候选 token 池和网络结构：

- 公共候选 token：16
- 图像候选 token：256
- 文本候选 token：64
- 每个实际传输 token：固定使用 16 个实信道符号
- utility weight：固定为 0.5
- 只改变图像与文本合计的实际传输预算

默认五组的实际分配为：

| 图文预算 | 公共 | 图像 | 文本 | 总传输 token | 信道使用量 |
|---:|---:|---:|---:|---:|---:|
| 208 | 16 | 168 | 40 | 224 | 3584 |
| 240 | 16 | 200 | 40 | 256 | 4096 |
| 272 | 16 | 232 | 40 | 288 | 4608 |
| 304 | 16 | 248 | 56 | 320 | 5120 |
| 320 | 16 | 256 | 64 | 336 | 5376 |

这些设置改变的是“送进信道的 token 数”，不会改变编码器生成的候选 token
数量、Transformer 的结构或解码器输出形状。因此原始 V6.1 权重可以严格加载，
但各组仍应使用相同数据量和训练轮数进行微调后再公平比较。

## 第一步：上传覆盖文件

把本实验工程中的以下文件/目录上传到服务器项目根目录并覆盖同名文件：

- `models`
- `utils`
- `analysis`
- `train.py`
- `test.py`
- `v61_gate_train.py`
- `v61_gate_eval.py`
- `v61_token_budget_train.py`
- `v61_token_budget_eval.py`

## 第二步：先做冒烟测试

```bash
cd /data1/zw/WYH/multimodel
export TOKENIZERS_PARALLELISM=false

python v61_token_budget_train.py \
  --config configs/v61_bandwidth.yaml \
  --budgets 208 \
  --checkpoint V6.1_model_weights \
  --output-dir experiment_results/token_budget_smoke \
  --seeds 42 \
  --smoke-test \
  --batch-size 2 \
  --num-workers 2 \
  --data-root /data1/zw/WYH/multimodel/COCO2017 \
  --vae-path /data1/zw/WYH/multimodel/pretrained/sd-vae-ft-mse \
  --text-model-path /data1/zw/WYH/multimodel/pretrained/clip-vit-base-patch32
```

日志开头应显示：

```text
[token budget 208] expected allocation C=16, I=168, T=40; channel_uses=3584
checkpoint exact=872 expanded=0 missing=0
```

## 第三步：训练五组控制变量

目前先用 seed 42，避免把随机种子实验混进来。五组都从同一个原始
`seed42.pth` 初始化：

```bash
mkdir -p logs

nohup python v61_token_budget_train.py \
  --config configs/v61_bandwidth.yaml \
  --budgets 208 240 272 304 320 \
  --checkpoint V6.1_model_weights \
  --output-dir experiment_results/token_budget_train \
  --seeds 42 \
  --utility-weight 0.5 \
  --epochs 2 \
  --train-samples 2000 \
  --val-samples 500 \
  --batch-size 2 \
  --num-workers 2 \
  --data-root /data1/zw/WYH/multimodel/COCO2017 \
  --vae-path /data1/zw/WYH/multimodel/pretrained/sd-vae-ft-mse \
  --text-model-path /data1/zw/WYH/multimodel/pretrained/clip-vit-base-patch32 \
  > logs/token_budget_train.log 2>&1 &
```

查看进度：

```bash
tail -f logs/token_budget_train.log
```

## 第四步：统一评估

```bash
nohup python v61_token_budget_eval.py \
  --config configs/v61_bandwidth.yaml \
  --checkpoint-root experiment_results/token_budget_train \
  --output-dir experiment_results/token_budget_eval \
  --budgets 208 240 272 304 320 \
  --seeds 42 \
  --utility-weight 0.5 \
  --snr-list 0 5 10 15 20 \
  --ablations full \
  --samples 500 \
  --batch-size 2 \
  --num-workers 2 \
  --data-root /data1/zw/WYH/multimodel/COCO2017 \
  --vae-path /data1/zw/WYH/multimodel/pretrained/sd-vae-ft-mse \
  --text-model-path /data1/zw/WYH/multimodel/pretrained/clip-vit-base-patch32 \
  > logs/token_budget_eval.log 2>&1 &
```

## 第五步：生成汇总 CSV

```bash
python analysis/summarize_token_budget.py \
  --input experiment_results/token_budget_eval/all_token_budgets.json \
  --output experiment_results/token_budget_eval/token_budget_summary.csv
```

最终下载：

```text
/data1/zw/WYH/multimodel/experiment_results/token_budget_eval/token_budget_summary.csv
```

## 结果解释注意事项

这五组同时改变了 token 总数和信道使用量，因此回答的是：

> 当每个 token 的表示宽度固定为 16 时，多传语义 token 是否值得？

它与已经完成的“固定 288 个 token、增加每个 token 的信道符号数”的带宽实验
互补。若之后需要严格区分“token 数”和“总信道使用量”的作用，再做总信道使用量
固定的二维对照。
