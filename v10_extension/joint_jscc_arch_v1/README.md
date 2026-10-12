# V10：图文 JSCC 架构改版 V1

状态：新增候选实现，不是已经验证有效的最优模型。旧源码、权重、结果和划分不覆盖。
本机无 Torch；实际梯度、预训练权重、GPU、FSO 代理适用性及恢复效果由服务器预检/实验验证。

## 本版做了什么

1. **整段文本压缩**：冻结 CLIP，按每段最多75个内容 token 加 BOS/EOS 编码；添加全局位置，再用32个可学习查询压缩全部已覆盖文本。默认总容量128 token（包含 BOS/EOS），不是无限长；超出部分明确标记截断。
2. **自回归文字恢复**：复用旧文字解码器初始化，使用词嵌入、因果掩码、EOS 与逐词生成。训练才使用正确前缀；评估只看已接收特征及自己的历史输出。EOS 与 PAD 同 ID 时以训练目标长度正确屏蔽 padding。不是增加原文 UTF8 旁路。
3. **JSCC 接口**：保留旧小型映射初始化，新增16x16图像 latent 的空间残差编码/恢复与两层跨 token 注意力残差映射，源端/信道映射/接收端共同带噪训练。固定 C32/I224/T32、每槽16实维、2304复符号，一帧发送一次。
4. **可靠性感知接收**：输入导频估计 SNR/可靠性、公开缺失位置、接收可用标记和观测幅度；不读取真实 SNR、原文、真实 CFO 或湍流状态。当前导频可靠性是帧级估计广播到各槽，位置差异来自公开缺失标记及观测，不冒充逐槽真实 CSI。旧7.5dB固定文字门控不进入新接收图。
5. **Common 辅助任务**：粗尺度4x4 latent 与短文本前缀恢复。辅助头只读物理接收 C 槽，位于私有槽混合之前；不额外发送一帧，也不把完整帧里的 C 诊断冒充独立低带宽 Common-only 传输。
6. **训练调整**：先干净恢复，再 FSO 带噪联合训练；VAE/CLIP 冻结，图像使用同一确定性 posterior mean。旧 latent 锚权重降为0.25，文字权重1.0，像素0.75、结构0.35，属于候选配置，不声称最优权重。没有新增模型下载、环境升级或自动长训练。

## 实验组与可解释范围

| 组 | 内容 | 主要比较 |
|---|---|---|
| B0_legacy | 冻结的旧 B2；同一确定性 VAE 输入；文字容量77 | 历史结构参考，不是新结构单项因果消融 |
| B1_text | 整段文本压缩、AR；旧映射与新固定长度接口 | 新文字/接口基础组 |
| B2_codec | B1 + 空间/跨 token JSCC + 可靠性接收 | B2−B1 是接口改动组合，不能独立归因于其中一个模块 |
| B3_common | B2 + Common 辅助损失 | B3−B2 衡量 Common 辅助训练 |
| B4_separate | 与 B3 同结构/参数/损失；阻断跨模态神经路径 | B3−B4 衡量联合处理；同参数规模，不声称等 FLOPs |

所有新组从完全相同初始张量开始，独立训练，不从上一组训练后的权重继续。
B4 的 C32 分为图像16、文本16；源 Transformer、TX/RX 跨 token 映射和接收 Transformer 使用块对角掩码，任务头只读取相应模态记忆。它仍共享同一帧的物理损伤，不是两条独立光链路。
只对 B1–B4 比较覆盖文本 WER；B0 覆盖77、新组覆盖128，不能把覆盖长度变化称为纯解码器收益。

默认：每训练组1000图文对，2轮干净+3轮FSO，共5000前向、2500优化更新；四训练组共20000前向。batch1、累积2、FP32、物理 GPU1。固定 seed42 只是复现标识，没有多种子扫描。
验证32对、每物理条件1个固定天气路径；clean +6物理点、5组，共1120条结果。短预检仅8训练对、每阶段1轮、2验证对、前2物理点。
主比较是 B3−B4 的低质量链路 PSNR/覆盖文本 WER；SSIM、完整原文 WER、双方可用、失败帧、干净校正增益及实际资源同时报告。所有区间仅对图片重采样，不覆盖训练/天气总体变异。历史 validation 不冒充最终独立 test。

## 光链路和代理门禁

保留已有完整 V3 相干 FSO：AWG64GSa/s、4SPS、16GBaud、RRC滚降0.1；湍流/指向损失与时变路径配置不改。最终评估调用完整波形链路，不用可微代理给自己打分。
训练复用完成的 V2 校准 bank，因其同公开物理配置。每组在初始、干净训练后、FSO训练后，使用当前模型的实际编码载荷，在公共 holdout 路径上新噪声全波形比较代理误差。图文样本仅来自 train，不使用 validation/test 校准。
任何门禁失败即停止、保留日志和已有阶段权重，不自动放宽误差门限。代理不是硬件可微模型。共同时间/峰值功率上限不代表实际平均光功率/能量严格相等，结果保留每次发射能量，不能宣称严格等能量优势。

## 1. 上传并解压

把 `v10_joint_jscc_arch_v1.tar.gz` 和 `.tar.gz.sha256` 上传到服务器现有 `/data1/zw/WYH/multimodel/`。
如果用终端上传，将“你的服务器地址”替换成现有 SSH 地址；也可以使用你已经连接的文件上传工具。不要新建并列项目。

```bash
scp v10_joint_jscc_arch_v1.tar.gz v10_joint_jscc_arch_v1.tar.gz.sha256 zw@你的服务器地址:/data1/zw/WYH/multimodel/
```

以下命令在服务器执行：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_joint_jscc_arch_v1.tar.gz.sha256
test ! -e _v10_joint_jscc_arch_v1 && tar -xzf v10_joint_jscc_arch_v1.tar.gz
"$PYTHON_BIN" -B _v10_joint_jscc_arch_v1/install.py --project-root .
"$PYTHON_BIN" -B v10_extension/joint_jscc_arch_v1/install.py --project-root . --verify-only
```

安装器核对历史依赖和新文件哈希；不一致时拒绝覆盖，不要自行加覆盖选项。
共享 HANDOFF 只追加新交接内容，追加前保存原始字节备份。旧权重、旧实验目录、数据不进入代码包。

## 2. 启动完整短筛选

启动前查看 GPU1 的可用显存；不足16GiB时先等待，不能停止其他人的任务。
预检/运行中采样自由显存不得小于4GiB。阈值是保护门禁，不是新模型已测峰值。

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used,memory.free --format=csv
(
  set -C
  nohup "$PYTHON_BIN" -B -u v10_extension/joint_jscc_arch_v1/launch.py pipeline \
    experiment_results/v10_joint_jscc_arch_v1_01 --max-hours 4 \
    > experiment_results/v10_joint_jscc_arch_v1_01.launch.log 2>&1 < /dev/null &
  echo $!
)
```

顺序：CPU/真实 Torch 合同 →8对 GPU 真预检（四新组）→当前载荷代理门禁→正式四组训练→完整波形验证→独立验收→自动收集。默认从旧完成 V2 B2 初始化，不从这次全解冻权重继续。
默认旧依赖路径：

```text
experiment_results/v10_fso_fluctuating_v2_01/v10_screen
experiment_results/v10_fso_fluctuating_v2_01/calibration
experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth
experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth
```

若实际完成的 V2 文件夹不同，用 `--source-run` 和必要的 `--calibration` 指向已完成目录；不要用只有轻量记录、没有大权重的解压目录。源码和权重身份仍会严格检查。
若输出名已存在，用 `_02` 等新名字，同时更改外层日志名。不要删除旧输出来绕过保护。
子 shell 的 `set -C` 只保护本次日志不被覆盖，不改变全局 shell 设置；`.launch.log` 与入口保留的 `.log` 名称分开，避免新创建的外层日志被误判成旧任务。
仅预检可将 `pipeline` 改为 `smoke`、选另一个全新输出目录；不会继续大筛选。

## 3. 查看进度和时间

```bash
tail -n 50 experiment_results/v10_joint_jscc_arch_v1_01.launch.log
tail -n 30 experiment_results/v10_joint_jscc_arch_v1_01/v10_screen.log
"$PYTHON_BIN" -m json.tool experiment_results/v10_joint_jscc_arch_v1_01/v10_screen/progress.json
pgrep -af 'joint_jscc_arch_v1/(launch|run).py'
```

前期只有外层日志和 `v10_smoke.log` 是正常的。`v10_screen.log` 在预检通过后才创建。
训练显示组、阶段、轮数、step、CE、像素损失和实测显存；逐词评估比旧并行文字头慢。
暂按约1–3小时规划，必须以真实预检和前100–200步更新；不能把旧版4.5分钟训练时间套到本版。默认4小时是总停止预算，不是保证完成时间，单组训练2小时、预检45分钟也都是停止预算。
保存了 clean/fso 两阶段新权重，但没有优化器/RNG精确续训协议。失败后不自动无限重试，不冒充无缝续训。

## 4. 完成后发回

自动生成：

```text
experiment_results/v10_joint_jscc_arch_v1_01/v10_joint_jscc_arch_v1_records.tar.gz
```

把这个小记录包发回即可：包含报告、全部指标、图文样例、星座图、实际 SNR/功率/时长/能量、训练/显存/代理适用性证据。
大权重、数据和 bank 留在服务器。若停止/报错，先发外层日志、最近的 `v10_smoke.log` 或 `v10_screen.log` 以及 `FAILED.json`；没有 COMPLETE 不能宣称完成。

报告入口：`v10_screen/report.html`；汇总：`v10_screen/summary.json`。Covered WER 和 Full-source WER 都是错误率，越低越好，不是“准确率”。模拟连续符号不捏造 BER。

## 文献采纳边界

- [DeepJSCC](https://arxiv.org/abs/1809.01733)：采纳带噪联合优化编码/恢复；不声称复现其 AWGN/瑞利曲线或必然胜过数字传输。
- [DeepSC 公开实现](https://github.com/13274086/DeepSC/blob/master/utils.py)：采纳 teacher forcing 与自回归实际解码；本版不复现其 Europarl 数据/MI损失/理想 CSI 假设。
- [DeepJSCC-OFDM](https://arxiv.org/html/2101.03909v1)：参考显式接收处理结合学习残差；本版仍为既有单载波相干 FSO，不是 OFDM。

本版回答“新接口能否改善恢复、联合处理有无额外收益”，不自动证明论文可发表、数字方案更差、Common-only低带宽可用、硬件真实性或所有长期全解冻训练无效。
