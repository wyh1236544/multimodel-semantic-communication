# DOCCI + Long-CLIP + C/I/T joint JSCC V8

这是独立、可撤回的新架构诊断版本，不覆盖 COCO/V4/V6/V7，也不声称更换数据和扩大图片一定有效。正式研究问题是：在相同 DOCCI 输入、符号预算、预训练骨干、注册参数量、训练步数和信道实现下，联合 Common/接收端融合是否比模态隔离多带来收益。

## 改了什么

- 数据只从 `DOCCI/docci_descriptions.jsonlines` 读取原始图文。官方 train=9647 用于训练，qual_dev=100 用于固定探索性验证；test=5000、qual_test=100 不参与模型输入、选择或训练。旧 `ln_v10_splits.json` 不变。
- 图片从 DOCCI 原始大图直接缩放到完整画面 512×512 letterbox，不中心裁剪、不随机翻转、不把历史128小图放大。冻结 SD-VAE 产生4×64×64 latent；新空间 JSCC 编码器下采样到32×32×8，接收端学习空间解码。指标仅评价完整画面的有效矩形，黑边不计入；矩形只用于损失/评价，不给 RX。
- 原 CLIP 的公共语义特征替换为真实 Long-CLIP-B（ViT-B/16），图像塔仍是原生224，**不是把 Long-CLIP 也改成512**。超过原生248上下文的文字切为连续246内容token块，保留全部有序特征，不仅做全局平均；这不是无限上下文 Long-CLIP。
- BART-base仍负责原文恢复。默认完整描述上限512 BART tokens（含BOS/EOS），可先审计后选择256/1024。超过上限会硬报错，不静默截断/过滤；CLIP与BART tokenizer计数不相同。文本私有通信层是64×64连续数，经学习的查询压缩、记忆重建，再用冻结 BART 自回归生成全文。
- Common不再沿用历史冻结投影：新图像/文本池化、32-slot Common压缩、RX融合和空间图像通信层一起训练。BART/VAE/Long-CLIP骨干仍冻结；这是训练通信层，不是全参数解冻。
- 分四阶段：完整长文本私有预训练 → 图像私有预训练 → paired clean joint/separate → 继续FSO带噪训练（25% clean anchors）。有非有限值就停，不悄悄跳过更新。
- 两个主组：`separate` 的Common前16仅图像、后16仅文字，图像/文本RX不互读私有分支；`joint` 的32 Common联合提取，双方RX融合接收到的另一模态私有特征。注册参数量相同，但实际注意力长度/计算量不同，报告明确记录。
- 每个已训练状态都评估 normal、zero_common、fusion_off、shuffle_common、shuffle_image、shuffle_text。错配来自另一条**接收到的**信号，不是干净原文/原图。控制组复用自己的同一RX帧，属于因果干预，不是省符号的新发送方案。

## 符号和链路预算

512主版本：C=32×16=512实数，I=32×32×8=8192实数，T=64×64=4096实数。合计12800实数，配对为6400复连续符号；仅发送一次R=1。256预检为3328复符号。它们不是“12800字节”，没有定义QAM比特映射，BER、比特率、字节压缩比不能从实数个数编造。

物理评价沿用经过历史验证的相干V3 waveform/DSP：AWG64GSa/s、4SPS、16GBaud、RRC滚降0.1；按新版实际负载重新计算导频、前导、保护、滤波尾和probe时间，不沿用旧3072槽。主组相同的是时间/峰值/硬件上限和天气、噪声种子，**不是相同实际波形或相同平均能耗**。保留实际导频SNR、接收/发射功率与能量、EVM、接收锁定、资源统计。

训练用可微Gamma-Gamma、指向衰落和复噪声符号代理。预设代理SNR=[5,10,15,20]，覆盖moderate/strong；代理不含完整时序天气、DAC、同步、MZM等。实际验证包含clean、weak_p06、moderate_p18、moderate_p21、strong_p18，FSO条件的名称是发射功率配置，不是最终接收SNR。主科学分析看FSO末态joint−separate的配对收益；clean与FSO训练总步数不同，不能把二者差异全归因于带噪训练。

默认screen训练128/验证16，仅一个种子；结果是探索性验证，不是独立测试或论文结论。此版本没有JPEG/FEC/QAM数字基线，也没有扩散模型；不能单凭此实验宣称连续优于数字，或新尺寸优于历史不同数据/分辨率。

## 先准备服务器离线文件

```text
/data1/zw/WYH/multimodel/
  DOCCI/docci_descriptions.jsonlines
  DOCCI/images_unpacked/.../train_00000.jpg （内部目录可不同，按唯一文件名索引）
  pretrained/bart-base/                    （已有完整encoder+decoder权重）
  pretrained/sd-vae-ft-mse/                （已有）
  pretrained/clip-vit-base-patch32/        （仅复用原CLIP tokenizer文件）
  pretrained/longclip-B.pt                 （新的599MB权重）
```

本地下载和安全解压见 [DOWNLOADS.md](DOWNLOADS.md)。不安装或升级Torch/Transformers/Diffusers，也不下载BART第二份；Long-CLIP实现和许可已随源代码打包。它只有核实过官方SHA256才允许反序列化。

## 上传、解压和安装（不启动训练）

把源码压缩包和 `.sha256` 上传到服务器项目根目录。DOCCI数据和Long-CLIP权重单独上传，不在源码包里。

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_docci_longclip_joint_v8.tar.gz.sha256
test ! -e _v10_docci_longclip_joint_v8
tar -xzf v10_docci_longclip_joint_v8.tar.gz --keep-old-files
"$PYTHON_BIN" -B _v10_docci_longclip_joint_v8/install.py --project-root .
"$PYTHON_BIN" -B v10_extension/docci_longclip_joint_v8/launch.py data-check
```

`install.py`先核对历史源码依赖和所有冲突，不覆盖已有不同文件；只新增模块并备份追加HANDOFF。重复解压必须停在已有 staging 提示处，安装相同版本可重复校验。历史手改依赖若不匹配，发报错，不要删除旧文件硬过。DOCCI完整标注计数在真实run必须符合官方9647/5000/100/100。

`data-check`输出全train/qual_dev的BART长度覆盖率、Long-CLIP原生248覆盖率及缺图。若配置512不能容纳选中的完整描述，改为 `--max-text-tokens 1024`，后面所有启动命令同样带该选项。不能拿被截断的目标评价全文恢复。

## 第一轮推荐：诊断，不要一上来长训

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
export CUDA_VISIBLE_DEVICES=1
"$PYTHON_BIN" -B -u v10_extension/docci_longclip_joint_v8/launch.py diagnostic \
  experiment_results/v10_docci_longclip_joint_v8_diag_01 --max-hours 4
```

启动器自动依次跑 `v10_smoke256`、`v10_smoke512`、`v10_diagnostic`，CPU合同测试也强制运行。两次真实预检都用TRAIN8/VAL2和各阶段8步，包括真实BART/VAE/Long-CLIP、512图像反向传播、checkpoint roundtrip、物理帧和接收边界检查，不能使用随机模型替代实际骨干。主诊断独立重新初始化，不复用SMOKE权重：TRAIN32/VAL8、文本400步、图像128步、每主组clean128+FSO64步。

文本预训练用完整原文与zero/shuffled记忆验证工程门：均值全文WER≤0.6、相对zero/shuffled改善≥0.05、自然EOS不碰长度上限。SMOKE只检查能执行，不判断恢复好坏；主诊断未过门则停止联合训练，留下 `QUALITY_STOP.json` 和长文本恢复页面，仍自动收集记录。不自动降低阈值，不能把它叫完成的联合模型。后续可以开**新目录**加文本训练：

```bash
"$PYTHON_BIN" -B -u v10_extension/docci_longclip_joint_v8/launch.py diagnostic \
  experiment_results/v10_docci_longclip_joint_v8_diag_02 --text-updates 2000 --max-hours 6
```

可选：如果你自己的 `P0_ce width64` 私有codec已有可信 `TRAIN_COMPLETE.json`，可显式 `--text-init /确切目录/last.pth` warm-start旧长度位置的前缀，新增位置和slots重新学习。默认不加载历史COCO checkpoint，不恢复旧优化器，不把旧短文本成功当新长文本成功。

诊断结果分析通过后再开screen：

```bash
"$PYTHON_BIN" -B -u v10_extension/docci_longclip_joint_v8/launch.py screen \
  experiment_results/v10_docci_longclip_joint_v8_screen_01 --max-hours 8
```

默认screen/pipeline的计划相同：TRAIN128/VAL16、文本1200步、图像512步、每组clean512+FSO256步。`pipeline`不是额外扩大数据或论文终版；扩大采样需新独立配置并保留其哈希。所有计划实际batch=1、梯度累积2、FP32；损失权重和proxy参数在config.json可见。

## 时间、显存、进度

这里没有实测Torch/预训练CUDA速度，因此不承诺小时数。`--max-hours`是**安全上限，不是预计耗时**，包括CPU测试、两次预检、缓存、训练、生成和物理评价。冻结特征按输入/原始模型完整哈希缓存，新目录可以复用，但训练状态不复用。512图像梯度和全文自回归比旧128/短描述慢，第一轮先诊断；看前20–100步ETA，再估计下一组。

启动要求至少16GiB空闲，循环保留4GiB空闲；这不是显存不会OOM的证明。不会清理或停止别人的GPU进程。预检保存峰值allocated/reserved GiB。若OOM发对应smoke日志，再按实测修改，不擅自降主实验分辨率。

```bash
tail -n 80 experiment_results/v10_docci_longclip_joint_v8_diag_01/v10_diagnostic.log
tail -n 80 experiment_results/v10_docci_longclip_joint_v8_diag_01/v10_smoke512.log
pgrep -af 'docci_longclip_joint_v8/(launch|run).py'
nvidia-smi -i 1 --query-gpu=name,memory.used,memory.free --format=csv
```

如果现有output/log已存在，换 `_02`，不要删除旧结果。若终端断线风险，可在你既有tmux会话运行；这里不安装新工具、不写crontab、不改变服务器环境。

## 发什么给我

正常结束或质量门停止后，启动器生成：

`experiment_results/v10_docci_longclip_joint_v8_diag_01/v10_docci_longclip_joint_v8_records.tar.gz`

上传这个包即可，保留完整权重在服务器。包里有完整长文本、512源图/恢复图、所有控制组指标、接收到的C/I/T帧、星座图、实际SNR和资源、训练梯度/更新审计、源代码快照和哈希，没有预训练模型/原始大图/cache。`actual_image_text.html`展示固定前两个验证样本，不按效果挑图。

手动重收集（避免PYTHON_BIN空变量）：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_longclip_joint_v8/collect.py \
  experiment_results/v10_docci_longclip_joint_v8_diag_01/v10_diagnostic \
  experiment_results/v10_docci_longclip_joint_v8_diag_01/v10_docci_longclip_joint_v8_records_retry01.tar.gz
```

异常退出没有有效终态时不要伪造COMPLETE，先发 `v10_smoke256.log` / `v10_smoke512.log` / `v10_diagnostic.log` 的末尾160行。评分失败、门停止和运行失败分开记录。
