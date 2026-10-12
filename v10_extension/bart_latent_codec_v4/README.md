# BART可解码压缩潜空间V4

状态：候选代码，不是文字恢复已解决。只新增本模块，旧V3/V2/C/I/T、图片、
权重、划分、结果均保留。不升级服务器环境，不下载新预训练模型，不启动
远程任务。使用现有完整`pretrained/bart-base`，本轮不加载CLIP/VAE或图片。

## 新结构与比较边界

文本 → 冻结BART encoder → 三层可学习查询压缩器 → T32×d连续潜变量 →
信道 → 三层重建器 → 32×768语义记忆 → 冻结BART decoder → 实际恢复文字。

不再四位置拼接，不展开回128个原词位置，不要求新记忆等于原词逐位置特征。
查询注意整句的有效源特征；源mask仅用于TX。每个潜变量槽L2归一化到平均
实数平方1。接收端固定公开all32 mask，EOS决定输出长度，不免费传原文长度。
预训练BART全部冻结，但解码计算保留到重建器的反向梯度；无LoRA/全解冻。
默认可训练约26M，真实数量由`trainability.json`记录；不是仅增加几个小层。

- P0_ce：最终原文交叉熵。
- P1_kd：同初始化/数据/优化更新/载荷，交叉熵＋0.5倍词概率蒸馏。
  同一个冻结BART的原生源记忆作为TRAIN教师；温度1，原文仍是主要标签。
  蒸馏额外前向算力单列，不宣称相同训练FLOPs。

32×64=2048实数/1024复符号；32×32=512复符号；32×16=256复符号。
这是特征/信道资源，不是UTF8字节压缩比。单帧R=1，无原文数字旁路。
32×64比旧V3窄分支占更多载荷，不能把性能变化直接宣称同资源架构收益。
新旧对照应固定容量/资源后再比较；本轮重点先验证可解码性。

## 首轮：只做无噪声诊断

默认8个原固定TRAIN IDs的COCO原始短caption，minimum_annotation_id政策，
seed42只作复现，不扫训练种子。BART cache max128，截断与覆盖原文都记录。
每组batch4×1200optimizer updates=4800样本呈现，FP32。不是旧V3的
1200forward/600updates。训练从第一步就用CE，无原位置MSE/特征预热。
AdamW1e-4/weight_decay.01，前100updates线性预热，再线性衰减到LR下限。

Native源记忆只作TRAIN诊断；实际原生cache/teacher与增量logits误差门禁。
8/8正确是期待目标；原门槛WER<=.20、零/外部错配记忆至少差.10、必须全部
自然EOS保持。全部实际greedy/zero/shuffled/beam4输出保存，不筛样例。
beam4＋no-repeat3＋penalty1.2仅辅助展示，不能替代无约束greedy的质量门禁。
质量失败输出overfit_stop/native_stop，流程完成标SCREEN_COMPLETE，不能
解释成模型恢复成功。异常/非有限梯度/冻结或重载失败立即停，无自动重试。

## 可选扩大与FSO：先看首轮，暂不运行

`pipeline`重新做诊断，只有各自过门槛的分支从未拟合8句的初始参数重新训练
1024train×3000optimizer updates；固定32validation，不按验证挑checkpoint。
各自验证WER<=.60、同记忆依赖/自然EOS门槛，才开始噪声实验：

从同一个干净checkpoint分成clean_continue和fso_continue，各1000updates、
同batch/data顺序/LR/教师/记忆一致性权重。FSO支路用.75噪声任务＋.25clean
anchor；clean支路对应完整clean任务。两个支路都用同一个冻结干净codec的
32×768记忆作一致性目标(.1权重)，不是强行匹配原128词的位置。

FSO训练为可微信道符号代理：Gamma-Gamma光强的平方根作相干场增益，指向
损失、随机相位，噪声固定在衰落前；32个导频估计CSI再LMMSE式均衡。默认
moderate/strong两组手设分布与-5/0/5/10/20dB轮换。参数在config.json。
此SNR是名义单位符号功率/噪声，不是光功率推导SNR；没有波形同步拒绝、
时域相关湍流、DAC/MZM、真实带宽，**不声称与完整物理信道分布一致**。
代理训练无同步失锁，本轮最终波形评价保留失锁、全部错误与实际发射开销。

每个clean/clean_continue/fso_continue在同32validation评价15个代理点和
原光配置全部6种完整波形。读取完成B4目录的channel_config，沿用真实
64GSa/s、4SPS、16GBaud、RRC.1及V3同步；发送本轮完整紧凑载荷，不切旧bank。
导频/probe/preamble/tails/guards全收费，单次发送，不把信道重复试验合并。
实际光功率、导频SNR、失锁、时长、能量、星座保存；analog BER/CRC为null。
相同采样上限/峰值包络不是严格相同实际能量，clean和FSO代理不能拿同dB
数字当完整波形等损伤。不训练图像，不输出图片PSNR，不证明图文联合收益。

## 上传、解压、安装

把两个发布文件先下载到本机，再在本机终端上传。替换SSH地址；这里不猜IP：

```bash
scp /你的下载目录/v10_bart_latent_codec_v4.tar.gz /你的下载目录/v10_bart_latent_codec_v4.tar.gz.sha256 zw@服务器IP:/data1/zw/WYH/multimodel/
```

服务器逐段执行，不复制提示符；任一步出错先停，不删除冲突对象：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_bart_latent_codec_v4.tar.gz.sha256
test ! -e _v10_bart_latent_codec_v4 && tar -xzf v10_bart_latent_codec_v4.tar.gz
"$PYTHON_BIN" -B _v10_bart_latent_codec_v4/install.py --project-root /data1/zw/WYH/multimodel
```

安装器逐文件校验已安装的V3及历史依赖；不同内容先拒绝，不能覆盖。
完全相同安装可重验。共享HANDOFF先按原字节备份，再追加新版本交接。
包为USTAR，不含macOS扩展属性、数据或权重。现有Py3.8/Torch1.12/
Transformers4.30.2不变；没有PEFT/einops等新依赖。

## 第一轮只运行这个：物理GPU1

```bash
"$PYTHON_BIN" -B v10_extension/bart_latent_codec_v4/launch.py data-check
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used,memory.free --format=csv
"$PYTHON_BIN" -B v10_extension/bart_latent_codec_v4/launch.py diagnostic experiment_results/v10_bart_latent_codec_v4_01 --width 64 --max-hours 2
```

启动至少16GiB空闲/运行保留4GiB，磁盘预留8GiB；不足停止，不杀别人进程。
自动先执行非Torch和10项Torch/HF小模型检查，再两句两组8updates实际smoke，
通过工程检查后做两组正式诊断。smoke质量不代表成功，也不初始化正式训练。
保守预计首轮20–60分钟（未实测），smoke约3–10分钟；2小时是停止上限，
不是预计时长。进度每20updates写入真实速度与本训练阶段ETA，不含后续生成。
本地缺Torch/真实预训练，所有真实学习和显存结论以服务器smoke为准。

前台与后台只选一种，不可重复启动。需要后台时：

```bash
mkdir -p experiment_results
test ! -e experiment_results/v10_bart_latent_codec_v4_01 && test ! -e experiment_results/v10_bart_latent_codec_v4_01_launch.log && nohup /home/zw/anaconda3/envs/torch/bin/python -B -u v10_extension/bart_latent_codec_v4/launch.py diagnostic experiment_results/v10_bart_latent_codec_v4_01 --width 64 --max-hours 2 > experiment_results/v10_bart_latent_codec_v4_01_launch.log 2>&1 &
```

查看：

```bash
tail -n 60 experiment_results/v10_bart_latent_codec_v4_01/v10_smoke.log
tail -n 60 experiment_results/v10_bart_latent_codec_v4_01/v10_screen.log
/home/zw/anaconda3/envs/torch/bin/python -m json.tool experiment_results/v10_bart_latent_codec_v4_01/v10_screen/progress.json
```

完成或质量失败都发这个包（权重留服务器）：

```text
experiment_results/v10_bart_latent_codec_v4_01/v10_bart_latent_codec_v4_records.tar.gz
```

执行异常尚无包时发smoke/screen日志、FAILED.json和learning_audit.json。
同名输出拒绝覆盖，重跑改02并保留01。不支持精确断点续训。

后续经首轮结果确认，再另开新目录扩大（会重新诊断，不是续跑）：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/bart_latent_codec_v4/launch.py pipeline experiment_results/v10_bart_latent_codec_v4_expand_01 --width 64 --max-hours 8
```

默认光配置来源`experiment_results/v10_joint_jscc_arch_v1_01/v10_screen`；不同
实际路径用`--channel-source-run`明确指定。可选`--width 32`或`--width 16`做
独立容量实验，使用不同输出名；本版不偷偷从宽模型截取权重。

## 文献与结论边界

这是根据[LD4LG论文](https://papers.neurips.cc/paper_files/paper/2023/file/b2a2bd5d5051ff6af52e1ef60aefd255-Paper-Conference.pdf)
和[作者语言自编码器](https://github.com/justinlovelace/latent-diffusion-for-language/blob/main/latent_models/perceiver_ae.py)
思想的独立实现，加入本项目通信约束与词级KD，不是逐项复现论文训练规模。
更多来源/实现差异见[SOURCES.md](SOURCES.md)。成功的小样本记忆不代表泛化，
单seed/历史validation不支持投稿成功率、连续胜数字、joint胜separate或硬件
结论。修好的T接回CLIP公共C与图像I、图文联合消融属于后续，不假装本轮完成。
