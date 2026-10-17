# V19：先比较，不再盲目加训

2026-10-17。新增实际数字链路与V18负SNR推理；没有新训练，也没有新图文桥。
完整约定见[PROTOCOL.md](PROTOCOL.md)，检查边界见[VERIFICATION.md](VERIFICATION.md)。

## 会做什么

- 复用V18已完成的三组0–20dB/clean输出，保留原始文件。
- 实际UTF-8压缩、CRC、受保护长度头、LDPC、QPSK、软译码、有界解压；不是用容量公式假装收发成功。
- 两种LDPC码率都报告，全部冗余和64导频+648帧头计入9216复符号；不按结果选赢家。
- 冻结原V14、V18 clean和V18 noise权重，补−10/−8/−6/−5/−4/−2dB，每档3次。
- 输出WER曲线、实际包成功率/BER/星座、100篇真实对比文字、30篇事实审核空表。
- 容量/压缩预算另列；事实质量PENDING，没有把WER改名为语义准确率。

不做：新训练、负SNR再训练、缩预算模型训练、图像训练/PSNR、FSO/光波形、速率硬件模拟、自动BERTScore/事实打分。

## 上传与解压（只增加新文件夹）

把源码包和.sha256上传到`/data1/zw/WYH/multimodel/`，然后在服务器执行：

```bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_docci_digital_compare_v19.tar.gz.sha256 &&
test ! -e v10_extension/docci_digital_compare_v19 &&
tar -xzf v10_docci_digital_compare_v19.tar.gz --keep-old-files

/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_digital_compare_v19/launch.py check
```

现有V18源码及其原历史依赖必须保持原样；缺失/漂移会停，不自动覆盖/下载。
预期原始运行：`experiment_results/v18_text_noise_01`；不是桌面的records轻包。
GPU补测还需要原`v14_text_pilot_01/main/text`特征缓存、三份保存权重和本地BART。
路径不同可用`--v18-root`，但权重身份与100篇描述不能偷换。

## 推荐：一次完成数字对照＋负SNR补测

启动前检查自己有权使用的GPU1；不停止其他任务。主程序内部自动先测试、CPU预检和真实保存权重锚点smoke。

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used,memory.free --format=csv

test ! -e experiment_results/v19_text_compare_01 &&
test ! -e experiment_results/v19_text_compare_01_launch.log &&
nohup "$PYTHON_BIN" -B -u \
  v10_extension/docci_digital_compare_v19/launch.py pipeline \
  experiment_results/v19_text_compare_01 --max-hours 4 \
  > experiment_results/v19_text_compare_01_launch.log 2>&1 < /dev/null &
```

数字主实验6800包；负SNR主推理5400次，不训练。启动GPU空闲16GiB、运行保留4GiB只是保护阈值，不是本版实测峰值。
按V18历史速度，负SNR推理约1–3小时是粗估，坏信道更易撞生成上限可能更慢；GPU新阶段未在本地执行，不能保证时间。上限4小时含5分钟收集预留，超时保留部分，不自动重跑。
CPU数字耗时以smoke/main日志实际ETA为准，不把笔记本合成测试速度当服务器正式耗时。

## GPU忙：先只做CPU数字实验

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
test ! -e experiment_results/v19_digital_01 &&
test ! -e experiment_results/v19_digital_01_launch.log &&
nohup "$PYTHON_BIN" -B -u \
  v10_extension/docci_digital_compare_v19/launch.py digital \
  experiment_results/v19_digital_01 --max-hours 2 \
  > experiment_results/v19_digital_01_launch.log 2>&1 < /dev/null &
```

此命令不导入Torch模型、不占GPU、不装依赖。状态DIGITAL_COMPLETE不是完整负SNR对照完成。
随后只补GPU，不重做数字主实验：

```bash
test ! -e experiment_results/v19_negative_01 &&
test ! -e experiment_results/v19_negative_01_launch.log &&
nohup "$PYTHON_BIN" -B -u \
  v10_extension/docci_digital_compare_v19/launch.py negative \
  experiment_results/v19_negative_01 \
  --digital-root experiment_results/v19_digital_01 --max-hours 4 \
  > experiment_results/v19_negative_01_launch.log 2>&1 < /dev/null &
```

此路径只读复制已完成数字派生记录；独立收集仍可CPU重放数字译码作核验，不是新的数字主实验。不要在已启动pipeline后再同时启动digital/negative。

## 查看进度与取回

```bash
tail -n 50 experiment_results/v19_text_compare_01_launch.log
tail -n 20 experiment_results/v19_text_compare_01/digital.log
tail -n 20 experiment_results/v19_text_compare_01/negative.log
```

后两项在对应阶段开始前不存在是正常的。`PIPELINE.json`显示阶段，最终以独立`REPORT.json`和外部records校验为准；看到生成结束不等于已收集完。
自动生成在运行目录旁边：

```text
experiment_results/v19_text_compare_01_records.tar.gz
experiment_results/v19_text_compare_01_records.tar.gz.sha256
```

把这两个发回；不要传权重、原生特征缓存或整个数据集，不要把PRIVATE records推到公开GitHub。
如只收集失败、训练/推理已完成，使用新归档名，不重复pipeline：

```bash
"$PYTHON_BIN" -B -u v10_extension/docci_digital_compare_v19/collect.py \
  experiment_results/v19_text_compare_01 \
  experiment_results/v19_text_compare_01_records_collect02.tar.gz \
  --max-minutes 30 --replay-digital
```

已完成report与封存行必须一致才允许再次收集，不悄悄改原始输出。没有准确恢复checkpoint的续跑功能。

## 如何解释结果

如果数字在0–20dB逐字恢复，照实报告，V18自适应收益仍成立但不等于资源优势。
若某负SNR连续WER较低，先看配对区间、事实审核及整篇CRC擦除限制；不能推广成胜过所有数字/渐进语义方案。
不得把理论容量、模拟CPU秒数、符号计数分别称为实测成功率、光链路时延、实际传输速率。
