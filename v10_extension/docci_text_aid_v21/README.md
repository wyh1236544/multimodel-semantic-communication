# V21：先诊断，再小实验

2026-10-18。V20已完整完成但未证明文本辅助；本版本不是重新长训。
详细对照与事前标准见[协议](PROTOCOL.md)。学校服务器执行，本机只做CPU测试。
不安装依赖，原始V20/Fix1/数据/权重/缓存/结果不修改。

## 1. 上传并解压代码

将`v10_docci_text_aid_v21.tar.gz`和同名`.sha256`上传到现有项目根目录。
只需新代码包，不要重新上传数据集或LPIPS资产。

```bash
cd /data1/zw/WYH/multimodel &&
sha256sum -c v10_docci_text_aid_v21.tar.gz.sha256 &&
test ! -e v10_extension/docci_text_aid_v21 &&
tar -xzf v10_docci_text_aid_v21.tar.gz --keep-old-files &&
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_text_aid_v21/launch.py check
```

必须已有`experiment_results/v20_text_guided_image_fix1_01`完整运行目录，包含
三份`train_*/last.pth`和`cache/arrays`，轻量records不能代替。
V14/V18父权重、BART/VAE、`_v20_lpips_assets`仍使用原位置。
原目录不同请统一加`--v20-root 实际目录`；权重/缓存哈希仍严格固定。

## 2. 现在先运行这一条：仅诊断，不训练

```bash
cd /data1/zw/WYH/multimodel
test ! -e experiment_results/v21_diagnose_01 &&
test ! -e experiment_results/v21_diagnose_01_launch.log &&
nohup /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_text_aid_v21/launch.py diagnose \
  experiment_results/v21_diagnose_01 --max-hours 2 \
  > experiment_results/v21_diagnose_01_launch.log 2>&1 < /dev/null &
```

入口固定物理GPU1，继承原Fix1真实CUDA门禁（起始空闲至少20GiB），运行中保留
4GiB GPU及项目磁盘；本版本项目磁盘起始要求8GiB。不停止他人任务，不降batch。
先CPU测试→严格CUDA/新神经合同→真实保存权重smoke→完整诊断→CPU验收→PRIVATE收集。
smoke先复现原V20同一权重的真实0dB输出锚点：TX及观测哈希必须一致，
PSNR/SSIM/LPIPS/MSE在1e−6容差内一致；不符直接停止，不跳过或自动放宽。
若任一步失败，保留失败并停止，不自动重试，不转入训练。

诊断主要是前向推理，暂按10–30分钟安排，**没有实测新GPU时间**，2小时只是硬预算，
含5分钟收集预留；第一次完整输入哈希审计和共享服务器负载会影响耗时。

查看总进度和实际阶段：

```bash
tail -n 60 /data1/zw/WYH/multimodel/experiment_results/v21_diagnose_01_launch.log
tail -n 30 /data1/zw/WYH/multimodel/experiment_results/v21_diagnose_01/main.log
```

`main.log`在CUDA检查和smoke结束后才出现，不出现不等于卡住。
完成标记是根`PIPELINE.json`的COMPLETE及main/VERIFIED.json，不仅是GPU进程消失。
完成后发`experiment_results/v21_diagnose_01_records.tar.gz`，不用发大权重。

## 3. 诊断后再决定：小规模接收融合训练

不要和诊断同时运行。此命令需用户决定后手动执行，不由上一步自动启动。

```bash
cd /data1/zw/WYH/multimodel
test ! -e experiment_results/v21_pilot_nominal_01 &&
test ! -e experiment_results/v21_pilot_nominal_01_launch.log &&
nohup /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_text_aid_v21/launch.py pilot \
  experiment_results/v21_pilot_nominal_01 --profile nominal --max-hours 4 \
  > experiment_results/v21_pilot_nominal_01_launch.log 2>&1 < /dev/null &
```

512对×8轮；三组各1024更新；只训练接收融合514018参数。图像codec/发送通路和
文本模型全部冻结。三组均从同一V20 A权重独立初始化，不串行继承训练。
可靠文本来自V18无噪声接收，是特权可行性诊断；真实带噪接收记忆另列结果。

规划约1–2小时，**仅粗估，未实测**；以各组`progress.json`的实际ETA为准。
硬上限4小时包含检查、smoke、训练、评价、收集，不保证主实验必然在预算内完成。
完成后发`experiment_results/v21_pilot_nominal_01_records.tar.gz`。

仅在分析nominal后另行批准：同一pilot命令换NEW输出和`--profile missing25`，
增加人工接收特征局部擦除。它不是FSO，不能和nominal分数混成一个结论。

## 4. 结果、失败与收集

主结果：`main/VERIFIED.json`，真实PNG网页：`main/index.html`。
diagnose详行：`main/rows.jsonl`；pilot：`main/{self,correct,donor}/rows.jsonl`。
训练日志为主`main.log`及各组`history.jsonl`，三组按顺序运行。
网页包含原图、各干预输出、SNR、PSNR、LPIPS；不包含虚构/美化的重建图。

若自动收集失败，在确认根PIPELINE已不为RUNNING后用NEW归档名手动收集：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_text_aid_v21/collect.py \
  experiment_results/v21_diagnose_01 \
  experiment_results/v21_diagnose_01_records_collect02.tar.gz
```

PRIVATE包包含实际DOCCI内容及图像，只发给分析者，不推送公共GitHub。
大数组/权重保留服务器。失败包可以收集，但明确FAILED/PARTIAL，不伪装完整成功。
原报告不覆盖；已存在输出、归档、源码哈希不符时停止，先发错误再处理。
