# V16：受保护的图像→文本接收语义桥

状态：2026-10-16 新实现，**尚未在真实服务器上训练或测量增益**。本机没有 Torch/Transformers；本地 CPU 合同和实际神经网络验证分开记录。不是已证明互促、优于数字通信或论文复现。

## 先做什么

先只跑真实 smoke，把轻量包发回。不自动启动 V14、不重复已有文本训练、不加载全量语言模型微调、不自动进入 FSO。所有命令从现有 multimodel 根目录执行，物理 GPU1，现有离线环境，无 pip 安装。新目录拒绝覆盖；与正在运行的 GPU 任务错峰。

源码包只新增 `v10_extension/docci_rx_bridge_v16/`，**直接解压，不需要 install.py**：

```bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_docci_rx_bridge_v16.tar.gz.sha256 &&
test ! -e v10_extension/docci_rx_bridge_v16 &&
tar -xzf v10_docci_rx_bridge_v16.tar.gz --keep-old-files

export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
"$PYTHON_BIN" -B v10_extension/docci_rx_bridge_v16/launch.py check
nvidia-smi -i 1 --query-gpu=memory.total,memory.used,memory.free --format=csv
```

先核对真实权重（必须是这两组固定末次权重，不猜最新目录）：

```bash
sha256sum \
  experiment_results/v14_text_pilot_01/main/text/released64/train/last.pth \
  experiment_results/v14_image_pilot_02/main/image/expanded/train/last.pth
```

期望分别为：

```text
3e6743eb1bcdbe4bed5db14ba9c9d5cb6d30cb2676066fc41c63fbb09ee3b000
95c05e3b374deae2a3bd10603b8e95ebbd464456f7581a054f2b80597be5923b
```

路径改变但内容不变时，可以显式用 `--text-checkpoint` / `--image-checkpoint` 指定；不同 SHA 不允许静默换权重。若想换成后续全量文本训练的权重，需要另立匹配实验，不在此运行中改起点。

### 1. 真实 smoke（现在建议只执行这一条）

```bash
test ! -e experiment_results/v16_smoke_01 &&
test ! -e experiment_results/v16_smoke_01_launch.log &&
nohup "$PYTHON_BIN" -B -u v10_extension/docci_rx_bridge_v16/launch.py smoke \
  experiment_results/v16_smoke_01 --max-hours 1 \
  > experiment_results/v16_smoke_01_launch.log 2>&1 < /dev/null &
```

它先运行真实 Torch/HF 小模型合同，再缓存8条TRAIN/2条qual_dev，四种辅助各做对齐和3次CE更新，并检查生成与基线。**smoke 权重绝不用于主实验**。1小时是停止预算，不是耗时预测；先预留约10–30分钟用于排期，真实耗时未测，不能保证。

```bash
tail -n 60 experiment_results/v16_smoke_01_launch.log
tail -n 40 experiment_results/v16_smoke_01/smoke_cache.log
tail -n 40 experiment_results/v16_smoke_01/smoke_train.log
```

以 `LAUNCH_FINISHED.json`、子阶段 `FINISHED.json` 和 CPU `VERIFIED.json` 同时存在且没有FAILED为完成依据。只有末条训练日志不代表评价完成。CPU验收/收集不再固定180秒强制截断。

完成后收集（不重新训练）：

```bash
"$PYTHON_BIN" -B -u v10_extension/docci_rx_bridge_v16/collect.py \
  experiment_results/v16_smoke_01/smoke_train \
  experiment_results/v16_smoke_01_records.tar.gz
```

发回这个 records 包及 `.sha256`，如失败则发主启动日志和对应子阶段日志末160行。失败日志/原目录保留，重试用 `_02`，不删除现场。

### 2. 完整冻结特征缓存（smoke确认后再运行）

```bash
test ! -e experiment_results/v16_cache_01 &&
test ! -e experiment_results/v16_cache_01_launch.log &&
nohup "$PYTHON_BIN" -B -u v10_extension/docci_rx_bridge_v16/launch.py cache \
  experiment_results/v16_cache_01 --max-hours 6 \
  > experiment_results/v16_cache_01_launch.log 2>&1 < /dev/null &
```

此独立流水会重新做一次小 smoke，再缓存全部9647官方TRAIN和100条重复使用的qual_dev，不读取test/qual_test像素或描述。顺序加载文本与图像主干，避免同时常驻全部骨干。缓存原始冻结特征而不是训练后桥的输出。全量稠密FP32理论张量约23GB，压缩占用依内容而变；启动要求项目磁盘至少64GiB可用，运行保留4GiB。缓存及权重留服务器，不用下载。

显存启动可用至少16GiB、运行全卡可用至少4GiB只是保护阈值，不是已测峰值，也不能保证没有OOM。对齐使用实际batch64（可在运行前的新配置中设32/128/256），**并非全BART batch64**；CE每次1篇、累积8篇。大batch会增加缓存临时RAM/GPU使用，不会自动降batch。

### 3. 理想视觉辅助 vs 同规模自模态适配（独立手动启动）

```bash
test ! -e experiment_results/v16_oracle_01 &&
test ! -e experiment_results/v16_oracle_01_launch.log &&
nohup "$PYTHON_BIN" -B -u v10_extension/docci_rx_bridge_v16/launch.py oracle \
  experiment_results/v16_oracle_01 \
  --cache experiment_results/v16_cache_01/main --max-hours 8 \
  > experiment_results/v16_oracle_01_launch.log 2>&1 < /dev/null &
```

两个新桥同初始化规则、全量相同图文对、2轮缓存对齐、3轮CE、相同更新/解码/固定末次权重。主干完全冻结。每组CE为3618更新、真实对齐302更新（batch64，2轮）；CE累计28941篇呈现。不是续训原V14，也不按qual_dev选最优checkpoint。

若验证结果正确辅助没有超过自模态和真实错配，就在这个配置停止，检查接口和拟合，**不能推论全部多模态恢复不可能**。oracle为“给定此连接器的理想输入诊断”，不是理论上界。2pp均值与置信区间下限>0是两个独立条件；不要求下限也大于2pp。

```bash
"$PYTHON_BIN" -B -u v10_extension/docci_rx_bridge_v16/collect.py \
  experiment_results/v16_oracle_01/main \
  experiment_results/v16_oracle_01_records.tar.gz
```

### 4. 实际接收图像辅助（oracle通过之后，仍须用户手动启动）

```bash
test ! -e experiment_results/v16_actual_01 &&
test ! -e experiment_results/v16_actual_01_launch.log &&
nohup "$PYTHON_BIN" -B -u v10_extension/docci_rx_bridge_v16/launch.py actual \
  experiment_results/v16_actual_01 \
  --cache experiment_results/v16_cache_01/main \
  --oracle experiment_results/v16_oracle_01/main --max-hours 8 \
  > experiment_results/v16_actual_01_launch.log 2>&1 < /dev/null &
```

只训练rx_clip/rx_i0，**不重复训练self**，复用同缓存、同预算的已完成self。主比较提前指定rx_clip；I0是探索性，不在结果出来后把赢家改为确认性主假设。原图版本和接收图版本分别训练，不把分布切换的失败误当可行性结论。

```bash
"$PYTHON_BIN" -B -u v10_extension/docci_rx_bridge_v16/collect.py \
  experiment_results/v16_actual_01/main \
  experiment_results/v16_actual_01_records.tar.gz
```

训练耗时尚未测量。缓存、CE、自由生成和独立验收分开记录；history中的ETA只预测剩余CE更新，不含后续生成/收集。不要拿旧V14耗时或max-hours当新架构的实测时间。初次返回smoke后再决定是否调整运行前的固定预算；修改配置需在新缓存、新实验开始前完成，不事后挑点。

## 架构与四个配置

```text
文本 → 原BART编码 → released64 TX → 固定18432实数 → 原RX codec → T0(9×32×768)
                                                                    ↓ 原BART文本注意力
图像 → 原VAE编码 → 固定I2 TX → 固定8192实数 → I2 RX → I0 → 原VAE → 接收图像
                                                     ↓               ↓
                                           rx_i0空间特征       冻结Long-CLIP patch特征
                                                     └──视觉桥64×256──┘
                                                                    ↓
                         BART末两层：原文本cross-attn → 新门控cross-attn → 原FFN → 文本
```

| 训练配置 | 新桥的辅助输入 | 角色 |
|---|---|---|
| self | 自己未融合的T0，固定池化成196槽 | 同规模、同激活路径的单模态适配器 |
| oracle_clip | 原图的冻结Long-CLIP patch | 未经传输、未计资源的理想诊断，非正式通信结果 |
| rx_clip | 实际重建图像的冻结Long-CLIP patch | 提前指定的实际接收主比较 |
| rx_i0 | 实际图像RX中间32×32×96特征，固定池化14×14 | 空间特征探索性对照 |

所有辅助统一196×768输入、64×256查询槽。I0使用无参数零填充，self使用无参数固定池化，所以新增可训练参数与桥内张量形状完全一致。按层形状计算新增6,916,866个参数（约692万），运行时还会独立核对，不是本地已完成Torch测量。**冻结Long-CLIP前处理的计算成本不相同**：rx_clip推理多一次视觉模型，I0/self没有，不能宣称全系统FLOPs相等。实际参数量由trainability记录；不靠参数名假装公平。

预训练BART和两条codec均冻结；新桥只以外部hook接在BART末两层旧encoder_attn_layer_norm之后、FFN之前，不替换原始文本memory，不改HF缓存结构。旁路关闭直接返回旧张量。初始tanh gate=0，注意力权重非零初始化，防止双重零初始化锁死学习。桥模块登记在独立模块中，不登记到冻结BART里。

空段门来自原文本接收通路对`[decoder_start=2,BOS=0]`之后EOS的概率，取`1-p(EOS)`软门，不读原文、真实段数、有效长度或源mask。报告空段误激活和非空段抑制。监督标签只进TRAIN损失/评价，不进接收器接口。

对齐的目标是接收T0经软内容权重池化后的固定随机投影，InfoNCE在实际跨样本batch上计算；不是使用原文教师的额外未收费传输。它只是连接器预热假设，不代表学到图文语义或保证有收益。

每组评价正确辅助、真实不同pair辅助、零辅助、旁路关闭；**自己的文本memory始终固定**。零输入保留全部注意力支持槽；不等同旁路关闭。逐样本保留输出，不禁止重复，不按结果筛图片。

## 成功标准与不包括什么

固定clean/SNR=null，图文共26624实数=13312复符号计数，R1；原图oracle不满足等资源通信。内部图像40是旧clean条件哨兵，不是40dB测试。本版无新Common、波形、导频、数字基线、噪声训练或FSO结论，也未实现文本→图像/双向循环/扩散/LPIPS。

数字筛查：正确辅助比同规模self的平均WER至少降低2pp，配对95%区间下限>0；正确比真实错配至少降低1pp且区间下限>0；首段token相邻重复和全文word相邻重复均不增、空段误激活0。配对区间仅覆盖这批样本、不覆盖训练随机性。最后还必须通过独立人工事实审核，物体/颜色/数量/关系确实改善，才能宣称语义互助；在此之前`full_semantic_success=false`，即使数字筛查过了也不改成成功。

本版WER保持旧口径，主值为逐描述normalized WER均值，另报corpus WER、重复、提前EOS、容量停止。容量停止不自动当作重复。qual_dev已反复看过，不是最终独立测试。Long-CLIP参与模型后不拿同族分数当唯一评价；BERTScore/事实审核继续用独立V15流程，但新输出尚未打分。

新records包括真实图像和文字展示用`records/index.html`，仅少量固定顺序样例，非选优。原文注明仅供评价。收集在服务器独立校验大缓存/权重哈希，下载包排除这些大文件；本地可用可信report的`verify(path, arrays=False)`重查轻量输出，**不能声称在本地重新推理或重验了已排除的数组**。所有records/描述/图片均PRIVATE，不上传公开GitHub。

## 来源与实现边界

- [Flamingo](https://arxiv.org/abs/2204.14198)：借鉴冻结语言模型中新门控视觉cross-attention，不移植其参数/数据规模。
- [BLIP-2](https://proceedings.mlr.press/v202/li23q.html)：借鉴查询桥接与分阶段学习，不声称复现其效果。
- [LLaVA](https://arxiv.org/abs/2304.08485)：借鉴冻结视觉特征与语言模型接口；本版不是LLaVA/BART现成预训练对齐模型。
- [Transformers 4.30.2 BART源码](https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/models/bart/modeling_bart.py)：核对旧cross-attention norm→FFN位置；数值因果/cache/beam合同仍须在服务器执行。

源文件和历史依赖由MANIFEST固定，Python3.8语法检查、源码包独立解压及冲突保护在本地验证。具体通过/跳过项见VERIFICATION.md，不把未执行的Torch测试或真实训练写成通过。
