# V17：按段视觉修复，隔夜clean/带噪四组对照

2026-10-16，新实现。真实GPU/恢复收益/耗时尚未测。完整设计和不能越界的结论见[PROTOCOL.md](PROTOCOL.md)；本地验证边界见[VERIFICATION.md](VERIFICATION.md)。

只新增`v10_extension/docci_segment_bridge_v17/`，不覆盖V14/V16或Fix1，不改原缓存/权重/结果，不需要install.py或新模型下载。既有学校离线环境Python3.8/Torch1.12/Transformers4.30.2；不pip安装、不升级。物理GPU1。

## 1. 上传与解压

把源码包和校验文件上传`/data1/zw/WYH/multimodel/`：

```bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_docci_segment_bridge_v17.tar.gz.sha256 &&
test ! -e v10_extension/docci_segment_bridge_v17 &&
tar -xzf v10_docci_segment_bridge_v17.tar.gz --keep-old-files
```

已有同名模块不要覆盖/删除；先确认是否已经解压。包为标准USTAR，无macOS扩展头。

## 2. 睡前一条隔夜流水线

确认有权使用GPU1、空闲显存至少16GiB，项目磁盘空闲至少64GiB；不满足先等待，不停止他人作业。保护阈值不等于新架构实际峰值。

```bash
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used,memory.free --format=csv
df -h /data1/zw/WYH/multimodel
```

然后启动：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python

test ! -e experiment_results/v17_overnight_01 &&
test ! -e experiment_results/v17_overnight_01_launch.log &&
nohup "$PYTHON_BIN" -B -u v10_extension/docci_segment_bridge_v17/launch.py pipeline \
  experiment_results/v17_overnight_01 \
  --cache experiment_results/v16_cache_01/main --max-hours 8 \
  > experiment_results/v17_overnight_01_launch.log 2>&1 < /dev/null &
```

它自动依次执行：CPU/真实Torch合同 → 只读验收既有全量缓存 → 真实4组小smoke → TRAIN-only原生BART教师与VAL观测准备 → **C、D、A、B**四组 → 各组独立CPU验收 → 自动轻量收集。

不重训V14、不重新生成原V16缓存、不使用smoke权重初始化主实验。8小时包含上述阶段并留15分钟收集；新架构尚未测速，不能保证四组都在8小时内完成。工程/资源异常停止、不自动重试；已完成组保留，partial/NOT_RUN不伪装成功。不自动进入actual/FSO。

只想先做真实小预检：把`pipeline`改成`smoke`，输出改为新的`experiment_results/v17_smoke_01`、日志改同名，`--max-hours 1`。之后正式pipeline仍会做自己的小smoke，避免未经验证地复用不同输入。

## 3. 正在做什么 / 如何看进度

```bash
tail -n 50 experiment_results/v17_overnight_01_launch.log
tail -n 30 experiment_results/v17_overnight_01/cache_audit.log
tail -n 30 experiment_results/v17_overnight_01/inputs.log
tail -n 30 experiment_results/v17_overnight_01/C_self_mixed.log
tail -n 30 experiment_results/v17_overnight_01/D_oracle_mixed.log
```

某个日志尚不存在意味着未到该阶段，不等于失败。`PIPELINE.json`记录每阶段RUNNING/VERIFIED/NOT_RUN/UNVERIFIED_PARTIAL；`*_verify.log`是独立验收；`progress.json`在各组目录内。ETA_this_arm_CE仅该组剩余CE，不包含评价和后续组。

四组每组默认2轮batch64对齐（302更新）＋6轮CE（micro1/累积8，7236更新），新桥参数按公式4,429,826，实际Torch会核验。老师和旧主干均冻结；没有全参数解冻。

## 4. 明早发什么

正常或预算停止后，自动生成：

```text
experiment_results/v17_overnight_01_records.tar.gz
experiment_results/v17_overnight_01_records.tar.gz.sha256
```

下载并发送这两个文件即可。大缓存、teacher/观测数组、epoch/last权重留服务器。记录包含真实图文、所有对照输出、日志、源码/输入身份和浏览页，属于PRIVATE，不能上传公开GitHub。

主日志最后会写`V17 COMPLETE`、`PARTIAL_BUDGET`或`FAILED`。只有COMPLETE且`COLLECTED.json`中complete=true，四组才算完整完成。预算/失败退出1也可能已经成功保存partial包；先发包和日志，不删除现场或重复开任务。

如果自动收集失败，而任务已停止，手动CPU收集到新文件名：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_segment_bridge_v17/collect.py \
  experiment_results/v17_overnight_01 \
  experiment_results/v17_overnight_01_records_collect02.tar.gz
```

不重新训练；CPU只验证已经标记VERIFIED的完整组，未完成产物只作为明确标识的排障材料保存。源码或缓存校验失败不得靠删检查绕过。

## 5. 怎么判断真的互助

正式生成评价固定clean/10/0dB，训练混合还包含20/5dB。仅文本TX符号加AWGN，接收codec和EOS门重新计算；图像仍是理想原图oracle，不是FSO或完整真实通信结果。

主问题D对C在0dB；次问题B对A在clean。每条文字固定自身接收信息，比较正确图像、5个不同真实错配图像、清零和关闭旁路。主指标WER与原口径一致，全部对照公开报告；人工事实审核未完成前full_semantic_success始终false。

本版是BLIP-2/ALBEF训练原则启发的独立小模块，不是其已训练模型的移植。旧V16失败结果原样保存，不改变原晋级门槛。源码/GitHub同步不包含原始DOCCI图文、结果包或权重。
