# V16 Fix1：保留已完成缓存，修复source_tokens验收字段

2026-10-16。用户返回`report.py → index_check → KeyError: source_tokens`。
根因已由原始源码确认：`cache.create`将原始BPE数写入index，却漏写baseline JSONL；独立report要求两者一致。
之前合成fixture含有此列，未忠实覆盖真实生产者schema。新增回归直接运行可信原生产表达式，先复现同一KeyError，再验证兼容修复。

## 重要边界

- 不编辑原V16源文件、MANIFEST、source_snapshot、baseline、index或缓存数组。不要手改JSONL或删除字段检查。
- 只在CPU验收器内存中补齐**缺失**的source_tokens，从哈希封存的原始index读取；源ID、文本、SHA、split都须一致。
- arrays=True时，原验收器仍先从9×64原始token帧核对真实长度；错的现有字段、错的index计数、篡改数组/源哈希仍报错。绝不根据模型生成文本估算长度。
- 不改变接收器输入、预测、损失、阈值、训练量、载荷或信道。原失败记录保留，不伪装成原流水成功。
- 验收通过后只新增原本应由验收器写出的`VERIFIED.json`派生回执，不覆盖已有不同回执。并发/部分写入失败保留，不自动删除。
- 新训练/归档明确附带fix源码快照和修复身份。旧V16原始源码仍固定在SHA256 `7ea2874353f748c4f7f9bbae8d95ef6475def1804e388d505976bcb1732c89eb` 的MANIFEST。

## 上传与安装

将`v10_docci_rx_bridge_v16_fix1.tar.gz`及`.sha256`上传原项目根目录；只新增旁边的fix1目录，不覆盖旧V16。

```bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_docci_rx_bridge_v16_fix1.tar.gz.sha256 &&
test ! -e v10_extension/docci_rx_bridge_v16_fix1 &&
tar -xzf v10_docci_rx_bridge_v16_fix1.tar.gz --keep-old-files

export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
"$PYTHON_BIN" -B v10_extension/docci_rx_bridge_v16_fix1/test_fix.py
```

## 从当前失败处继续（不重新生成缓存）

如果原来按说明启动的是`experiment_results/v16_smoke_01`，执行：

```bash
test ! -e experiment_results/v16_smoke_fix1_01 &&
test ! -e experiment_results/v16_smoke_fix1_01_launch.log &&
nohup "$PYTHON_BIN" -B -u v10_extension/docci_rx_bridge_v16_fix1/launch.py resume-smoke \
  experiment_results/v16_smoke_01/smoke_cache \
  experiment_results/v16_smoke_fix1_01 --max-hours 1 \
  > experiment_results/v16_smoke_fix1_01_launch.log 2>&1 < /dev/null &
```

若原目录名不同，只替换第一个路径为实际`smoke_cache`。程序先检查原cache完整、必须是smoke而非全量，再运行CPU合同、原Torch/HF合同、重新CPU验收缓存；通过后仅在新目录运行尚未开始的四臂短训练/评价。原cache不重算、原V14不重训。
如果原目录已经有`smoke_train`，本入口拒绝重复训练：请先回传它的状态/日志，而不是重启。已有进程是否仍在运行由用户核实；程序不查杀其他作业。

```bash
tail -n 60 experiment_results/v16_smoke_fix1_01_launch.log
tail -n 40 experiment_results/v16_smoke_fix1_01/cache_verify.log
tail -n 40 experiment_results/v16_smoke_fix1_01/smoke_train.log
```

没有改显存保护：GPU1启动可用16GiB、运行可用储备4GiB。1小时是停止预算；耗时未实测。校验失败不会自动重训，保留原始异常。

## 完成后收集

以后用fix入口，不要再次直接调用原有report/collect（旧schema未改）：

```bash
"$PYTHON_BIN" -B -u v10_extension/docci_rx_bridge_v16_fix1/entry.py collect \
  experiment_results/v16_smoke_fix1_01/smoke_train \
  experiment_results/v16_smoke_fix1_01_records.tar.gz
```

将这个records及校验文件发回。归档不含大数组/权重，但有真实图文，仍为PRIVATE，不上传公开GitHub。

只想CPU复核缓存、不启动训练时：

```bash
"$PYTHON_BIN" -B -u v10_extension/docci_rx_bridge_v16_fix1/entry.py report \
  experiment_results/v16_smoke_01/smoke_cache
```

后续新阶段仍须手动启动：将原命令中的`docci_rx_bridge_v16/launch.py`替换成`docci_rx_bridge_v16_fix1/launch.py`，其他参数保持原协议。它只把原launcher的run/report子进程指向已记录的兼容入口；不自动执行cache/oracle/actual或重启旧任务。

## 验证状态

本地11项新CPU回归：真实生产表达式缺字段复现、原文件逐字节不变、全部四臂存储矩阵/收集、归档修复来源、错误字段与错误源长度拒绝、源哈希拒绝、上下文恢复、回执冲突保护、恢复配置一致、实际CPU报告子进程的恢复编排、拒绝重复训练。编排测试的GPU训练子进程用明确的TEST_ONLY存储fixture替代，不是实际训练证据。原V16的13项CPU回归也执行。Python3.8语法、源码USTAR/SHA/独立解压验证。
无Torch/Transformers，本地未重跑真实模型/GPU；服务器真实cache完整性仍待执行兼容验收，不能从这个字段报错推断恢复质量。
