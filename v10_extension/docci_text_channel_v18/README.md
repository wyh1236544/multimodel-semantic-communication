# V18：文本codec带噪训练，保留V14结构与所有旧结果

2026-10-17。当前是已实现、待服务器真实smoke的新实验，不是已取得抗噪收益。完整[冻结协议](PROTOCOL.md)，[本地验证边界](VERIFICATION.md)。

这次只改变训练方式：V14 released64的发送压缩器、接收重建器一起带噪训练，BART冻结。没有新CLIP、图像桥、Common或模型下载。继续使用旧2048条训练描述，避免数据量和噪声同时变化；不重训图像、不重复生成缓存。

## 1. 上传、解压（项目根目录）

把代码包与.sha256上传到 /data1/zw/WYH/multimodel，然后执行：

```bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_docci_text_channel_v18.tar.gz.sha256 &&
test ! -e v10_extension/docci_text_channel_v18 &&
tar -xzf v10_docci_text_channel_v18.tar.gz --keep-old-files
```

目录已存在不要删除或覆盖，先确认是否已经安装。标准USTAR，无macOS扩展头，不需要install.py或安装依赖。

## 2. 必须保留在服务器上的输入

```text
experiment_results/v14_text_pilot_01/main/text/
  backbone.json
  source_identity.json
  cache_train_identity.json
  cache_validation_identity.json
  cache_train/*.npz
  cache_validation/*.npz
  released64/train/last.pth
  released64/validation/rows.jsonl
pretrained/bart-base/
```

仅记录包不够，旧原生BART特征缓存必须仍在。启动会逐一核对固定元数据/数组与权重身份，不缺失时自动重建或下载。如果移动了完整text目录，可显式传 --features 新的项目内路径；目录结构和内容必须一致。

## 3. 一条流水线：预检通过后才训练

先确认自己有权使用GPU1，空闲显存至少16GiB，项目磁盘空闲至少16GiB；不要停止其他人的进程。

```bash
cd /data1/zw/WYH/multimodel
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used,memory.free --format=csv
df -h /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python

test ! -e experiment_results/v18_text_noise_01 &&
test ! -e experiment_results/v18_text_noise_01_launch.log &&
nohup "$PYTHON_BIN" -B -u v10_extension/docci_text_channel_v18/launch.py pipeline \
  experiment_results/v18_text_noise_01 --max-hours 8 \
  > experiment_results/v18_text_noise_01_launch.log 2>&1 < /dev/null &
```

不需要先重复运行其他V14/V16/V17训练。流水线先执行CPU/真实Torch合同、旧权重8TRAIN/2VAL小预检（两组各2更新、三条路径各12条生成），全通过后才正式运行。只想做预检，把pipeline改smoke、输出/日志换成新的v18_smoke_01、max-hours设1；正式pipeline仍有自己的独立预检。

默认每组2560更新（10轮、micro1/累计8），两个视图分别重算；正式三条路径在clean与0–20dB生成、每档3噪声实现，并有错配/清零诊断。原V14冻结基线不是新训练组。

预计整轮约3–5小时，8小时是硬上限而非保证完成。估计依据旧V14训练速度，未实测新路径；预检后的进度才可用于调整预期。freeGPU保护不是OOM结论。

## 4. 进度

```bash
tail -n 50 experiment_results/v18_text_noise_01_launch.log
tail -n 20 experiment_results/v18_text_noise_01/train_clean_continue.log
tail -n 20 experiment_results/v18_text_noise_01/train_noise_replay.log
tail -n 20 experiment_results/v18_text_noise_01/eval_noise_replay.log
```

日志还未出现代表还没到相应阶段。初始CPU缓存检查有每100条进度，不占GPU。PIPELINE.json记录各阶段，*_verify.log是独立验收。ETA仅对应当前训练组/评价，不是整条流水的ETA。后台启动终端不输出正常，先看launch.log，不要盲目重开。

## 5. 完成后发什么

```text
experiment_results/v18_text_noise_01_records.tar.gz
experiment_results/v18_text_noise_01_records.tar.gz.sha256
```

包内有真实原文与恢复文本、WER/SNR曲线、日志、所有对照和独立验收。属于PRIVATE，不能上传公开仓库。模型和大缓存留服务器。只有COMPLETE且COLLECTED.json.complete=true才是完整主实验，效果是否通过另看固定筛查与事实审核。

如果仅收集失败，确认所有相关进程已经停止后，用新文件名重收，不重训：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_text_channel_v18/collect.py \
  experiment_results/v18_text_noise_01 \
  experiment_results/v18_text_noise_01_records_collect02.tar.gz
```

部分训练/评价不会伪装完成。没有自动精确断点续训，不要用中间文件冒充last。缺缓存、哈希不符、Torch合同失败都先保留日志，不装包、不改旧文件绕过检查。
