# CLIP + BART + VAE 图文 C/I/T 候选 V1

新增模块，不覆盖任何旧代码、权重、结果或数据划分。没有启动服务器训练。
本地没有 Torch/Transformers/GPU；本地合同测试不代表模型有效，实际权重、梯度、显存和恢复质量由服务器门禁确认。

## 结构与边界

- 文本：完整离线 `facebook/bart-base` 编码器/解码器，保留预训练共享词嵌入、LM输出头和EOS起始约定，不再用 CLIP 词嵌入初始化一个随机语言解码器。容量固定128个 BART token，超出部分标记截断，并同时评价完整原文与覆盖原文。
- 文本私有：128个有序隐藏向量，每4个位置折叠为一个，压成 T32×16。接收端恢复固定128位置记忆，再接入完整 BART 解码器。原文掩码/长度只在发送或训练监督使用，不传给接收端。
- 图像私有：精确加载已完成 B4 分离模型，复用其图像 I224×16，添加轻量可训练TX/RX残差；接收端的历史图像基座只看接收图像私有槽，公共语义另经新残差分支进入图像恢复。VAE posterior mean固定，不重新随机抽 latent。
- 公共：BART文字特征、图像私有特征及 CLIP 图像/文字投影特征，经注意力压成 C32×16。CLIP文字按≤75内容token分块，不扩大77位置表。公共辅助损失只读取物理C槽，不从私有特征倒灌。
- 传输：C32+I224+T32，每槽16实维，合计4608实值/2304复符号，单次R=1。没有UTF8文本旁路。C两半、I、T分别归一化，分离控制不存在跨模态功率归一化旁路。
- 接收：只读符号、导频估计SNR/可靠性、可用标记和公开布局；BART记忆采用固定公开全1掩码，不知道源文本长度。实际评估是贪心自回归，不使用正确前缀；失败帧输出空文本/零图片并计入指标。
- 训练：首版冻结 BART、CLIP、VAE、历史图像基座，仅训练新压缩、公共融合、TX/RX和接收适配模块。冻结解码器仍能把梯度传回其输入，代码没有在这条训练路径使用no_grad/detach。

“恢复出的文字再辅助图片”的二次条件生成分支在本版**明确延期**，不混入第一次对照；图片已经可以使用收到的共同C。C/I/T是设计用途，不是已经证明严格语义解耦。本版也没有独立Common-only短帧或传统JPEG/QAM的新对比。

## 三个同结构、同符号资源候选

|候选|变化|回答的问题|
|---|---|---|
|J0_no_clip_input|移除 CLIP 源端特征输入；保留同一CLIP训练辅助教师|传入CLIP图文特征是否有额外收益，不是完全无CLIP系统|
|J1_joint|完整图文公共融合与共同C接收|主候选|
|J2_separate|C前16仅图像、后16仅文本；源端注意力与接收任务记忆隔离|相同参数登记规模/符号预算下，联合路径是否有收益|

所有候选从相同新桥接初始张量开始；每个候选先干净训练，然后从自己的同一干净权重分叉 `clean_continue` 与 `fso`，更新次数/样本顺序一致。
比较登记可训练参数、资源与实际能量；不声称有效活跃参数、FLOPs或实际能量严格相等。旧CLIP文字基线与BART的tokenizer/容量/模型规模不同，不冒充单因素因果比较。

## 四道门禁

1. 服务器13项小型Torch/BART合同，随后真实BART/B4/VAE前向与输入梯度预检。
2. 未压缩完整BART实际恢复作为TRAIN诊断；T32文字私有压缩进行8段TRAIN文本、1200前向更新，其中前200步特征暖启动。每2次前向一次优化，共600优化更新。检查正常/置零/错配接收记忆的真实生成。
3. 仅当TRAIN覆盖WER≤0.20，置零与错配分别至少恶化0.10时才允许继续。干净联合验证的覆盖WER≤0.60且满足记忆依赖；这是探索门禁，不是论文质量标准。失败保存 `overfit_stop` / `clean_validation_stop`，不自动加长训练或放宽门限。
4. 完整FSO使用既有V2公共校准bank；在当前编码载荷上做新噪声完整波形holdout检查，通过才允许带噪训练，训练后再次检查。最终结果全部由V3完整波形链路评价，不用代理给自己打分。

第一轮建议只跑 `diagnostic`。它会先自动短smoke，再做1200步文字筛选；**不做联合/FSO训练**。`diagnostic_ready`只表示训练集文字桥接通过门禁，不代表泛化或FSO训练完成。

完整 `pipeline` 默认：256训练图文对，每候选4轮干净训练、两分支各2轮继续，共3×(1024+512+512)=6144前向，另有1200文字诊断；验证16对、6物理条件+clean、6分支，共672条恢复结果。固定seed42仅为复现身份，不做多种子扫描。不使用test或业务calibration图像。

## 离线权重要求

服务器 `pretrained/bart-base/` 至少有 `config.json,pytorch_model.bin,vocab.json,merges.txt`；推荐同时上传之前下载的 `tokenizer.json,README.md,SHA256SUMS`。
预期下载版本为 `facebook/bart-base` 的 `aadd2ab0ae0c8268c7c9693540e9904811f36177`，FP32权重约558MB。程序记录实际本地文件哈希并检查已有SHA256SUMS，但不把本地哈希当作远程版本认证。

`pretrained/clip-vit-base-patch32/` 必须包含完整CLIP图像/文字权重和 `preprocessor_config.json`；仅旧文字塔权重不够。
若只有处理配置缺失，在联网本机下载该小文件，再上传到已有CLIP目录；不要在服务器升级环境：

```bash
curl --fail --location 'https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/preprocessor_config.json' --output clip_preprocessor_config.json
scp clip_preprocessor_config.json zw@你的服务器地址:/data1/zw/WYH/multimodel/pretrained/clip-vit-base-patch32/preprocessor_config.json
```

本版完全离线加载、不安装包、不下载权重。沿用Python3.8/Torch1.12/Transformers4.30.2/Diffusers0.20.2；若实际环境不同会记录，失败不要擅自升级。

## 上传与安装（只在现有multimodel项目内）

上传 `v10_bart_clip_joint_v1.tar.gz` 和同名 `.sha256` 到服务器项目根。已有BART权重也应按之前步骤上传解压到 `pretrained/bart-base`，不要误把代码包当权重包。

```bash
scp v10_bart_clip_joint_v1.tar.gz v10_bart_clip_joint_v1.tar.gz.sha256 zw@你的服务器地址:/data1/zw/WYH/multimodel/
```

服务器执行：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_bart_clip_joint_v1.tar.gz.sha256
test ! -e _v10_bart_clip_joint_v1 && tar -xzf v10_bart_clip_joint_v1.tar.gz
"$PYTHON_BIN" -B _v10_bart_clip_joint_v1/install.py --project-root .
"$PYTHON_BIN" -B v10_extension/bart_clip_joint_v1/launch.py check
```

安装先核验历史依赖和新源码哈希；不一致就停止，不覆盖。共享HANDOFF先保存原始字节备份再追加。旧text_jscc_v1/fix1与所有历史实验不变。
默认必须保有已完成 `joint_jscc_arch_v1_01/v10_screen` 的B4权重及 `fso_fluctuating_v2_01/v10_screen` 的B2权重，而不只是解压的轻量记录。路径不同用 `--source-run --legacy-run --calibration --checkpoint --a-delta --bart-path` 指定，身份仍严格核对。

## 先启动文字筛选：物理GPU1

先确认GPU1至少16GiB空闲；运行保留4GiB自由显存，阈值不是实测峰值保证。不动别人的进程。

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used,memory.free --format=csv
(
  set -C
  nohup "$PYTHON_BIN" -B -u v10_extension/bart_clip_joint_v1/launch.py diagnostic \
    experiment_results/v10_bart_clip_joint_v1_diagnostic_01 --max-hours 2 \
    > experiment_results/v10_bart_clip_joint_v1_diagnostic_01.launch.log 2>&1 < /dev/null &
  echo $!
)
```

查看：

```bash
tail -n 40 experiment_results/v10_bart_clip_joint_v1_diagnostic_01.launch.log
tail -n 40 experiment_results/v10_bart_clip_joint_v1_diagnostic_01/v10_diagnostic.log
"$PYTHON_BIN" -m json.tool experiment_results/v10_bart_clip_joint_v1_diagnostic_01/v10_diagnostic/progress.json
```

第一步smoke期间只有 `v10_smoke.log` 是正常的；完成后自动生成：
`experiment_results/v10_bart_clip_joint_v1_diagnostic_01/v10_bart_clip_joint_v1_records.tar.gz`。
发回这个小包即可，里面有真实文字样例、正常/置零/错配对照、CE与特征损失曲线、参数登记、显存和权重加载/梯度证据；大权重留在服务器。
如果报错，发外层日志、当前子日志、FAILED.json以及learning_audit.json，不用上传模型大权重。

## 通过后才跑完整对照

在全新输出目录，把 `diagnostic` 改为 `pipeline`，预算建议 `--max-hours 6`。它重新从同一初始化进行诊断/训练，不从记忆8句话的权重继续、不把旧diagnostic目录当resume。可以将上面命令两处输出名改为 `v10_bart_clip_joint_v1_screen_01`。
主日志在 `v10_screen.log`，结果可视化在 `v10_screen/report.html`；记录包名不变，目录不同。

粗略规划：短smoke约2–10分钟，文字筛选约15–60分钟，完整筛选约2–6小时；**都未在实际GPU上计时**，I/O、显卡共享和逐词生成会明显影响。日志每20步给本训练阶段的实测剩余时间，不包括加载/其它训练组/生成评估。2小时或6小时是保护上限，不是ETA保证；超时保留记录，不无限自动重试。

报告同时保留图像PSNR/SSIM、原文/恢复文本、覆盖/全原文WER、CER、BLEU2、截断/容量停止、导频SNR、物理噪声投影SNR、功率分项、调度时长、实际能量、星座图与失败帧。连续特征没有二进制BER/CRC，相关字段为null，不捏造准确率。

## 文献与兼容来源

采用BART完整预训练去噪编码器/解码器和冻结骨干适配思路，不声称复现某篇论文的FSO曲线：
[BART](https://aclanthology.org/2020.acl-main.703/)、[CLIPTrans](https://arxiv.org/abs/2308.15226)、[BLIP-2](https://proceedings.mlr.press/v202/li23q.html)。
CLIPTrans不是本项目的权重来源，也不引入其定制Transformers环境。本版用已安装HF接口传入 `encoder_outputs`，恢复增益、联合收益和低SNR优势必须由实际数据回答。
