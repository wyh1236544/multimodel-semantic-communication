# DOCCI independent reconstruction V10

新增 `docci_reconstruction_v10`，不覆盖 V9 或更早的源码、数据、检查点和结果。
本阶段先修复独立的图像/文字通信，不宣称 Common 或图文联合收益。
固定物理 GPU1、FP32、既有离线环境，不安装依赖、不下载骨干、不停止他人进程。

## 本次可检验的问题

| Arm | 实际更改 | 512画布下连续复符号数 | 初始化 |
|---|---|---:|---|
| I0_latent | 原V9空间网络；latentMSE +0.1 pixelMSE | 4096 | 认证的V8图像分支 |
| I1_pixel | 同一网络；pixelMSE +0.01 latentMSE | 4096 | 与I0完全相同 |
| I2_shifted | 64/96/128多尺度、交替移位窗口、相对位置偏置；pixel优先 | 4096 | 明确新初始化 |
| T0_whole | 原文512容量一次压成32×64 | 1024 | 认证的V6文本codec |
| T1_chunks | 原始BART BPE按62个内容token分段，每段含BOS/EOS，总64；共享codec | 9216 | 相同codec非位置参数；复制旧位置前64 |

I1−I0是匹配初始化、参数、样本、顺序、学习率、优化步数和资源的**损失函数对照**。
I2−I1资源/样本/优化步数一致，但新网络参数数和初始化不同，不能声称纯粹的架构因果收益。
T1−T0同时改变分段和载荷（9倍），是“降低单段压缩难度”的容量诊断，不是同资源压缩优势。
记录配对变化和小样本条件bootstrap，不覆盖训练随机性/天气总体或独立test。

I2是自行实现的Swin启发的**VAE潜空间**codec，不是SwinJSCC复现，仍冻结完整VAE。
只有发送端内部、接收端内部的多尺度连接，没有跨信道的干净特征旁路。
另记录两种加权损失在重建latent上的梯度范数；不把标量损失占比当总参数梯度占比。
不加载CLIP/Long-CLIP，不用扩散修图；后续独立分支可靠后再训练Common。

## 全文与帧协议

原始完整DOCCI描述先BART分词，不截断、不按长度剔除。512容量意味着最多510内容token。
固定 `ceil(510/62)=9` 段，**空段也编码、发送、解码和计费**；不需要可变长度/段数头。
一次完整固定帧R=1，不是全文重复9次。接收端只看到观测连续符号、可用性、导频SNR/可靠性和公开布局。
没有原文、原始长度、真实段数、源paddingmask、干净BART特征或UTF8旁路。
每段贪心生成，真实EOS停止，不强制EOS、不禁止重复、不用beam修饰主结论。
去除各段生成的特殊token后按原顺序拼BPE，再整体解码；不人为加入空格。
空段若幻觉了词也照样进入全文WER，不能根据原始段数删除。

对照输出：整个原生BART记忆参考；额外原生分段BART参考（明确ORACLE，源mask仅此诊断可用）；
初始codec、实际训练后、相同收到帧上的正常/清零/其他样本打乱。
图像有效全视图letterbox512，指标与loss排除黑边；有效矩形不进入接收模型。
无信道clean中的40dB只是有限条件哨兵，不是测量SNR。

## 上传、解压、安装（全部在multimodel内）

先下载本次源码包和附带`.sha256`，通过原来连接服务器的上传工具把两者放到
`/data1/zw/WYH/multimodel/`。**代码包不是数据包，不需要重新上传DOCCI或Long-CLIP。**

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_docci_reconstruction_v10.tar.gz.sha256
test ! -e _v10_docci_reconstruction_v10 && \
  tar -xzf v10_docci_reconstruction_v10.tar.gz --keep-old-files
"$PYTHON_BIN" -B _v10_docci_reconstruction_v10/install.py --project-root .
```

先确认hash检查输出OK，再解压/安装。若暂存目录已存在，这个test会阻止重复解压；
只有确认是同一个已校验包才重复运行安装器。安装器全量检查旧源码依赖和新目标冲突，
先字节备份HANDOFF再追加；同内容重装不改写，未知改动/符号链接拒绝。不要删除旧目录。
根目录旧run_v10.sh保持原样；必须用下面的新版直接入口。

## 第一轮推荐：五组独立clean筛查，不自动扩大

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
export CUDA_VISIBLE_DEVICES=1
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used,memory.free --format=csv
"$PYTHON_BIN" -B v10_extension/docci_reconstruction_v10/launch.py diagnostic \
  experiment_results/v10_docci_reconstruction_v10_diag_01 \
  --branch both --max-hours 4
```

全量源码校验 → 13个非Torch合约 +10个真正Torch/HF CPU合约 → 每个网络真实离线骨干/GPU预检
（文字512；三种图像网络256后512） → 新进程重置初始权重训练；绝不复用smoke权重。
所有请求的预检先完成，再进行第一组实际训练。
这轮**真的使用64个不同TRAIN原文/图片**，不是只有4个样本重复；固定qual_dev16评价，
固定TRAIN前8仅为训练恢复gate，不是唯一训练集。文本各1200优化步/batch1，
图像各512优化步/microbatch1/累积2；五组顺序执行，一组质量不好仍会保存并执行其他组。
工程错误/超时/显存越界则终止并保留日志，禁止自动重试或忽略非有限梯度。
完成的SMOKE、CLEAN_READY、QUALITY_STOP均清楚标注；验收成功不等于恢复成功。

也可先分两次缩小排查，每次必须新输出：

```bash
"$PYTHON_BIN" -B v10_extension/docci_reconstruction_v10/launch.py diagnostic \
  experiment_results/v10_docci_reconstruction_v10_image_01 --branch image --max-hours 2
"$PYTHON_BIN" -B v10_extension/docci_reconstruction_v10/launch.py diagnostic \
  experiment_results/v10_docci_reconstruction_v10_text_01 --branch text --max-hours 2
```

预计全五组约**30–120分钟，仅规划范围、V10未实测**；长文本实际贪心评价、加载和hash检查也耗时。
4小时为停止预算，不是ETA。启动要求空闲16GiB并保留4GiB，512真实反向预检决定能否继续，
不能保证“16GiB必然足够”。日志逐20步给本训练阶段ETA及峰值allocated/reserved；不含后续组/全部生成。

## 扩大与带噪（第一轮分析后，由你显式选择）

`screen`仍同样64 TRAIN/16 VAL，但文本3000步、图像1024步，从初始权重重训。
可复制config到**新文件**增大train_samples（最多2048）/val_samples（最多qual_dev100）。
不要覆盖发布config或根据test调参。没有自动选择validation最佳checkpoint或自动promote。

```bash
"$PYTHON_BIN" -B v10_extension/docci_reconstruction_v10/launch.py pipeline \
  experiment_results/v10_docci_reconstruction_v10_fso_01 \
  --branch both --max-hours 6
```

只有**每组自身**clean的TRAIN/validation gate都通过才分叉同一clean权重，
做相同步数/样本/LR的clean_continue与FSO_continue；其他失败组保留QUALITY_STOP。
冻结TRAIN清洁接收特征仅作loss目标，不能进入RX。
带噪训练为现有Gamma-Gamma/指向衰落/导频估计**符号级近似**，随机名义SNR−5/0/5/10/20，
不是全波形/真实设备可微训练。最终用完整V3光DSP模拟clean+四种FSO条件，保留同步拒绝和失败。
AWG64GSa/s、4SPS、16GBaud、RRC0.1、导频/探测/前导/尾部/保护间隔全部计费，真实功率/能量/导频及
接收投影SNR/时间记录。光学传输时间不含神经推理或CPU仿真运行时间。
连续特征没有天然二进制BER/bitrate；不捏造。相同公共峰值限制不是相同实际发射能量。
`pipeline`会重新训练，不是续跑已有diagnostic/screen；不自动执行。

## 显式资源扫参（暂不建议一次全跑）

`rate-info`只打印资源，不加载模型、不训练；`--image-width 8/16/32`分别4096/8192/16384复符号。
文字 `--text-width 16/32/64` 的九段载荷分别2304/4608/9216。
改变宽度必须显式fresh初始化，从同样新初始化训练比较，不能跨shape加载旧权重或宣称固定资源收益。

```bash
"$PYTHON_BIN" -B v10_extension/docci_reconstruction_v10/launch.py rate-info
"$PYTHON_BIN" -B v10_extension/docci_reconstruction_v10/launch.py diagnostic \
  experiment_results/v10_docci_reconstruction_v10_rate8192_01 \
  --branch image --image-width 16 --image-init fresh --max-hours 2
```

## 进度与发回结果

```bash
tail -n 40 experiment_results/v10_docci_reconstruction_v10_diag_01/diagnostic_T1_chunks.log
find experiment_results/v10_docci_reconstruction_v10_diag_01 -maxdepth 2 \
  -name 'run_identity.json' -print
nvidia-smi -i 1
```

当前组名字会出现在启动时Owned PID/log行；某个日志还不存在，表示未轮到那一组。
结束自动生成以下小包，发回**这一个**即可，不需大权重：
`experiment_results/v10_docci_reconstruction_v10_diag_01/v10_docci_reconstruction_v10_records.tar.gz`。
每组actual_recovery.html和总index.html可手动打开查看完整文本、失败、对照图片；所有16样本数值保留，
固定前4保存PNG，不挑好看样本。源码快照、实际优化/梯度/显存/资源和配对报告均在小包内。
若工程失败，先发当前`.log`及子目录FAILED.json，不能删FAILED伪造完成。

本地检查证据见VERIFICATION.md；论文依据与不可直接照搬的实验条件见SOURCES.md。
