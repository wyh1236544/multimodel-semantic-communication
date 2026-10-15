# DOCCI semantic probe V12

状态：研究代码交付；本地 CPU 合同/记录验收/安装恢复测试，不代表服务器真实模型预检通过或性能改善。
新增隔离模块，历史 V6–V11 源码、数据、权重、失败记录、结果和质量门槛不改写。
固定物理 GPU1，原有离线 Python3.8 / Torch1.12 / Transformers4.30.2 / Diffusers0.20.2，不装依赖。

## 本轮只回答两个问题

| 任务 | 初始化与训练 | 固定资源 | 能说明什么 |
|---|---|---|---|
| B0_ce | V11 累计20480步末次权重，NEW AdamW，纯CE再4096步 | 九个公开段、9216复载荷符号、R1 | 匹配的继续训练对照 |
| B1_ce_kd | 与B0相同；CE+0.5×完整词表KL蒸馏，温度2 | 与B0完全一致 | TRAIN原生BART分布监督是否改善接收memory/自由生成 |
| A0_image_only | 固定既有I2编码器/解码器和VAE，仅训练三层残差模块2048步 | 图像4096复载荷符号、R1 | 等规模接收适配器的无文本基线 |
| A1_oracle_text | 与A0相同参数、初始化/顺序/更新/LR，额外使用原始文本Long-CLIP向量 | 图像预算相同；文本未经过信道、未计费 | 理想语义辅助是否值得进一步研究；不是实际联合传输 |

固定2048条官方TRAIN及100条qual_dev；qual_dev已被反复用于研究诊断，不是独立测试。
smoke仅8 TRAIN/2 qual_dev、两步；主实验新进程重载原始批准权重，不读smoke检查点。
两组同一个seed42仅用于复现，不跑多种子大矩阵；固定末次权重，不按validation挑最佳。

文本只用冻结BART；原始memory与真实前缀只进入TRAIN损失或明确的事后诊断，不能进入实际RX。
纯CE对照完全不计算教师。KL使用全部词表并按有效token平均，PAD不贡献损失。
不叠加scheduled sampling/unlikelihood，不改贪心解码，不强制EOS、不加重复惩罚。
九段包括空段全部发送/解码/计费；没有免费真实长度、源mask或UTF8旁路。

图像使用512完整视图letterbox，原有VAE仍冻结，无扩散。新增3个空间调制残差模块，共399843个可训练参数；
每组均训练image-derived self-context、空间mask和调制网络，只有A1把文本向量加到同一context。
初始残差严格为零，从而初始输出等于原始I2。A0即使收到不同文本参数也必须完全忽略。
冻结Long-CLIP-B按全部连续246正文token窗口编码，再作长度加权归一化池化；不截断为77或248。
这里采用全描述池化向量，不声称保留了每个token的词序/精确字面信息。

图像两组均交替训练clean和25%实坐标置零，固定来源无关的训练/验证不同随机损伤；所有8192实坐标照常计费。
25%擦除是**人为接收特征诊断**，不是天气/FSO，也没有额外传递擦除mask或信道真SNR。
评价同一份存储观测：原始I2、重训A0、A1正确/打乱/清零文本；打乱只是同权重干预，不冒充另训无文本基线。
图像PRIMARY为mask25的PSNR；同时报告clean、SSIM、每例与配对区间。只在两个primary比较的CI下界均>0且clean平均不降时
标记PROMISING_DIAGNOSTIC_ONLY；否则NEEDS_REVIEW。这是新的筛查决策，不替换旧模型QUALITY_STOP门槛。
LPIPS明确NOT_MEASURED，没有安装包或下载感知骨干。不能宣称复现论文的感知质量/HiFiC/GAN收益。

## 上传与安装

从聊天下载源码包及sha256，使用原来的文件上传工具，把两者放到 `/data1/zw/WYH/multimodel/`。
不需要重新上传DOCCI、BART、VAE或Long-CLIP。包内没有权重、数据或完整实验结果。

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_docci_semantic_probe_v12.tar.gz.sha256
test ! -e _v10_docci_semantic_probe_v12 && \
  tar -xzf v10_docci_semantic_probe_v12.tar.gz --keep-old-files
"$PYTHON_BIN" -B _v10_docci_semantic_probe_v12/install.py --project-root .
```

必须先看到哈希OK再执行解压/安装。暂存目录已存在则不重复解压；不要删除旧目录。
安装器先检查全部历史依赖和新目标冲突；对既有HANDOFF作精确备份后追加，已安装同版本可重复核验。
不同内容/符号链接/旧源码漂移会停止，不能覆盖绕过。默认批准的真实权重：

- 文本：`experiment_results/v10_docci_text_continue_v11_20261014_201245/checkpoint_20480/last.pth`
- 图像：`experiment_results/v10_docci_reconstruction_v10_diag_01/diagnostic_I2_shifted/train_clean/last.pth`

权重及元数据SHA锁定在parents.json，先验SHA通过后才加载pickle；缺失不会退回随机权重。
同一个批准run若移动过，可用`--text-parent`/`--image-parent`指向真实run目录，但哈希必须完全一致。

## 推荐运行：先预检再主实验，GPU1顺序执行

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
export CUDA_VISIBLE_DEVICES=1
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used,memory.free --format=csv
"$PYTHON_BIN" -B v10_extension/docci_semantic_probe_v12/launch.py check
```

启动前要求16GiB可用并保留4GiB；只是安全下限，真实512图像反向smoke决定能否继续，不保证16GiB足够。
不得停止别人的进程。默认路径使用你已经上传的DOCCI根目录、pretrained/bart-base、pretrained/sd-vae-ft-mse、
longclip-B.pt与pretrained/clip-vit-base-patch32；不会访问网络。

先单独跑smoke（预计数分钟，非实测；30分钟是停止预算）：

```bash
"$PYTHON_BIN" -B -u v10_extension/docci_semantic_probe_v12/launch.py smoke \
  experiment_results/v10_docci_semantic_probe_v12_smoke_01 --task both --max-hours 0.5
```

确认SMOKE_VERIFIED.json存在并且没有FAILED.json后，再运行主实验。`run`会再次强制预检，随后另起主训练进程：

```bash
export V12_OUT=/data1/zw/WYH/multimodel/experiment_results/v10_docci_semantic_probe_v12_01
test ! -e "$V12_OUT" && test ! -e "${V12_OUT}_launch.log" && \
  nohup "$PYTHON_BIN" -B -u v10_extension/docci_semantic_probe_v12/launch.py run \
  "$V12_OUT" --task both --max-hours 6 > "${V12_OUT}_launch.log" 2>&1 < /dev/null &
```

四组顺序执行，不会同时占用两套大模型。首轮建议规划约1–3小时，**新V12未实测、不是承诺**；
依据仅为历史V11纯CE约0.106秒/步、旧I2含像素反向约0.484秒/步，蒸馏、2048图像缓存和长文本自由生成会额外耗时。
6小时是总停止预算；20步一条ETA只覆盖当前训练组，不包含后续生成、其他组和CPU收集。
若只想先做其中一项，分别使用`--task text`或`--task image`，每次全新输出，不能复用同一目录拼成一次both。

## 查看进度与结果

```bash
tail -n 50 /data1/zw/WYH/multimodel/experiment_results/v10_docci_semantic_probe_v12_01_launch.log
tail -n 40 /data1/zw/WYH/multimodel/experiment_results/v10_docci_semantic_probe_v12_01/main.log
find /data1/zw/WYH/multimodel/experiment_results/v10_docci_semantic_probe_v12_01 -name progress.json -print
```

main.log不存在表示预检尚未结束，不是丢日志。GPU_PROGRAM_FINISHED只表示子程序结束；
最外层LAUNCH_FINISHED.json表示请求的GPU流程结束，仍需独立CPU验收。任何异常停止、保留日志，不自动重试/缩batch。
QUALITY_STOP依旧可收集；不能删除FAILED来伪造成功。

训练完，单独执行CPU收集，不会重训或改写原结果，也没有旧180秒report超时：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_semantic_probe_v12/collect.py \
  experiment_results/v10_docci_semantic_probe_v12_01 \
  experiment_results/v10_docci_semantic_probe_v12_01_records.tar.gz --max-minutes 30
```

发回`v10_docci_semantic_probe_v12_01_records.tar.gz`即可。若收集中断，保留旧压缩包/contents，改新目标名`...records_collect02.tar.gz`，无需重训。
解开结果包后index.html展示全部100条原文/CE/KD恢复与固定前4图像的所有条件，不按效果选样；verified_analysis.json给出配对结果。
未保存PNG的图像仍全部保留数值。验收包括完整矩阵、源码/源身份、实际更新/损失/顺序、共享损伤、字面指标重算、
FP32前缀聚合边界与PNG子集MSE复核；不是重新进行GPU推理或全样本SSIM/LPIPS重算。

## 决策与暂不做的事

文本优先看匹配CE对照后的WER/CER、EOS/截断、TF正文CE和自由生成，不只看loss；旧门槛原样报告。
图像必须同时优于重训A0和打乱文本，且clean退化须公开；正确文本优于错误文本本身不够。
本轮不训练Common、不传输文本辅助信息、不做FSO、不做数字基线、不宣称压缩/净码率收益。
若理想辅助有稳定信号，下一阶段才换成真正收到的语义/文本并计入C+I+T和物理开销；不能把本轮原文当实际RX输入。
失败只否定这个有限预算/冻结主干/具体模块配置，不证明CLIP或图文互促原则不可能。
