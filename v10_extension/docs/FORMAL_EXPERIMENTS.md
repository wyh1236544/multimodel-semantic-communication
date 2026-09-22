# V10 正式论文实验执行说明

状态：实验框架已实现；真实服务器训练尚未执行。本文档不构成性能提升证据。

## 原则

冻结 V10-A 方法，不再叠加新模块。正式证据链分为：环境与划分验收、三个独立训练种子的核心模型、三个种子的关键消融、validation 决策冻结、一次性 test、跨种子与图像层级统计。test 不参与阈值、超参数、模式或论文主张选择。

## 运行顺序

从 `/data1/zw/WYH/multimodel` 执行，先只读检查 GPU，再逐阶段运行：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

export CUDA_VISIBLE_DEVICES=1
export FORMAL_TAG=paper_v10_02_noamp
bash run_v10_formal_suite.sh preflight
bash run_v10_formal_suite.sh train-core
bash run_v10_formal_suite.sh train-ablation
bash run_v10_formal_suite.sh eval-validation
```

正式入口固定使用 FP32（`--no-amp`）。首次服务器正式 mixed 训练在 AMP 下于 epoch 1、step 140 出现非有限梯度；失败目录 `v10_formal_paper_v10_01` 必须保留。无 AMP 版本使用新的 `paper_v10_02_noamp` 标签重新执行，不能覆盖或续用失败目录。

检查 validation 报告，确定主张、阈值和所有展示规则后，冻结协议：

```bash
bash run_v10_formal_suite.sh freeze
bash run_v10_formal_suite.sh eval-test
bash run_v10_formal_suite.sh aggregate
```

每阶段都会拒绝覆盖既有目录。中途失败时保留现场，并更换 `FORMAL_TAG` 或在明确记录原因后为单项创建新的 attempt 标识；不得删除失败目录后伪装成首次运行。

## 规模与含义

- 核心训练：mixed、AWGN-only、FSO-only，各 3 个训练种子。
- 关键消融：无 UEP、无接收适配器、无 Common 辅助任务，各 3 个训练种子。
- validation：500 个独立 image ID，每条件 5 次信道重复。
- test：1000 个独立 image ID，每条件 5 次信道重复，只能在冻结后运行。
- Common 因果对照：正确、批内错配、随机载荷，保持相同 Common 物理载荷。
- 统计：每个图先聚合信道重复，再进行跨图与跨训练种子的层级配对 bootstrap。

默认完整矩阵计算量很大。先以 `preflight` 的实测耗时估算总预算；若单 GPU 资源不足，不得静默缩小正式样本后仍称为完整实验。可以把不同种子分配到有权使用的不同 GPU，但同一输出目录只能由一个进程负责。

## 尚未自动纳入的强基线

传统分离式编码（图像压缩 + 信道编码/调制）和外部 DeepJSCC/ADJSCC 复现尚未接入，因为必须先明确实现、码率、功率、时延和许可证，不能用不公平的占位实现冒充强基线。完成核心矩阵后再作为独立工作包接入。

## 发回用于论文分析的文件

发回每个正式评估目录中的 `summary.json`、`samples.jsonl`、`run_identity.json`、`COMPLETE.json`，以及两个 `aggregate_test_*.json`。权重、COCO 数据和缓存不上传 GitHub。
