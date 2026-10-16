# 下一步：V14文本先跑完，V15评价并行准备

2026-10-16。V14校准和图像pilot02已回传并独立核验；图像两组不用重跑。
同100篇qual_dev，理想clean/SNR=null：旧I2 20.0913dB，新64图22.9772dB，新2048图23.4150dB；原生VAE26.2312dB为非等资源参照。
图像每组5120新更新，训练两组约87分钟、main整体约95分钟，单训练种子。

**当前没有V14文本legacy64/released64新训练完成的证据。**先只读查状态，若任务已经运行，不重复启动：

```bash
cd /data1/zw/WYH/multimodel
pgrep -af 'docci_baseline_v14/(launch|run)\.py' || true
find experiment_results -maxdepth 2 -path '*/v14_text*/LAUNCH_FINISHED.json' -print
nvidia-smi -i 1 --query-gpu=index,memory.total,memory.used,memory.free --format=csv
```

确认没有自己正在跑的相同文本任务，物理GPU1至少16GiB空闲且有权使用后，再启动**新目录**。如果`v14_text_pilot_01`已存在，用`_02`并一起修改所有日志/收集路径，不删除旧目录。

```bash
cd /data1/zw/WYH/multimodel
test ! -e experiment_results/v14_text_pilot_01 &&
test ! -e experiment_results/v14_text_pilot_01_launch.log &&
PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python \
nohup /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_baseline_v14/launch.py train \
  experiment_results/v14_text_pilot_01 --task text --profile pilot \
  --max-hours 8 > experiment_results/v14_text_pilot_01_launch.log 2>&1 < /dev/null &
```

两组从各自预定的新初始化开始：2048篇，20轮，microbatch1/累积8，5120更新/组；同载荷9216复连续符号/篇、R1。仅codec可训练，BART冻结；纯CE，无CLIP/Common/噪声/扩散/KD。本轮只检查公开代码位置表示约定是否影响优化，不宣称新架构必胜。

预计1—3小时只是此前工程估算，**V14文本本版尚未实测**；8小时是停止预算。真实ETA按main日志，训练ETA不包括缓存、双解码/全对照评价、落盘和独立验收。运行保留4GiB整卡空闲，失败不绕过保护、不停止别人任务。

```bash
tail -n 40 experiment_results/v14_text_pilot_01_launch.log
tail -n 40 experiment_results/v14_text_pilot_01/main.log
```

main日志尚未出现时看smoke.log。外层LAUNCH_FINISHED.json才表示CPU合同、真实smoke、main和独立验收均已结束；最后update不等于全部完成。质量差也原样保留，工程通过不代表质量通过。

完成后：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_baseline_v14/collect.py \
  experiment_results/v14_text_pilot_01/main \
  experiment_results/v14_text_pilot_01_records.tar.gz --max-minutes 30
```

发回小records包，不发权重/全数据。并行能做的是V15只读评价和人工事实清单，不同时另开GPU训练。

数据量扩展、LoRA、InfoNCE、句子分段、长度头、带噪训练均是**后续待设计的独立实验**，本轮未实现/未启动。先根据V14验证质量与V15事实丢失类别选一项；扩大数据要匹配更新/token预算，不把同轮数当同预算。最终FSO/数字对照与图文互促仍需单独实验。
