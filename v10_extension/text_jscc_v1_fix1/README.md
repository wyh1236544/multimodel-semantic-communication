# V10：文字单模态JSCC V1 Fix1

## 本次故障修复

原服务器8项Torch测试通过，但真实GPU预检在native_overfit时报“Required source/compression/TX/RX/decoder did not learn”；GPU空闲46.09GiB，不是此次报错的显存原因。源码发现AR解码器复制自冻结旧模型，deepcopy保留冻结标记，候选解码器未显式解冻。原小型测试未冻结参考模型，遗漏这个初始化场景。

本版仅解冻新候选language副本，原CLIP/B4/VAE仍冻结不变；不移除学习审计、不降低质量/代理门禁、不增加训练来绕过故障。测试改为冻结参考初始化，并新增2项副本可训练/无共享旧权重/真实解码器梯度更新检查，共10项Torch合同，必须服务器重新执行。训练开始记录trainability.json，结束即写learning_audit.json，若失败明确输出哪一组没有梯度或更新。

新源码在text_jscc_v1_fix1，原text_jscc_v1与experiment_results/v10_text_jscc_v1_01都保留。按下面新包/新入口/新输出启动，不继续旧失败进程，不改旧MANIFEST。81>77是整句tokenizer提示；CLIP前向已有75内容token分块，本次RuntimeError不是位置索引越界，不要因此截短原文。修复后GPU有效性仍须真实预检，不能从静态检查保证通过。

状态：新增候选实现，尚未测得真实文字恢复收益。只改新增模块，不替换正式图文模型，保留旧权重和结果。原Python/PyTorch环境、物理GPU1、FP32；不安装依赖、不下载新模型、不停止外来进程。

## 新文字分支

现有tokenizer -> 可训练token嵌入/两层Transformer -> 48槽查询压缩 -> 16维连续映射 -> 接收映射/公开导频条件 -> 自回归文字解码。现有CLIP词嵌入经过旧投影初始化可训练词表，不把冻结CLIP整句特征当作唯一可恢复表示。旧CLIP/B4保持冻结供诊断/对照。源编码、压缩、TX、RX、生成头均审计真实梯度和参数更新。

只读取Localized Narratives文字标注，不打开/编码图片，不计算图片PSNR。为了精确加载历史对照仍会初始化冻结VAE/旧图文网络，**不是VAE参与训练**。

各训练阶段加入相同的4SPS/RRC载荷峰均比软约束（8dB目标、权重0.0005），避免仅在理想符号域改善而恶化光调制削顶；这是软损失，不是PAPR严格上限，最终仍以全波形测量为准。

现有CLIP tokenizer规范大小写/空白，容量128token含BOS/EOS，截断明确报告，**不是字节无损**。保留完整原文WER、覆盖文本WER、CER、BLEU2、原文/真实恢复句子、EOS/容量停机、逐句/语料WER。评估端CLIP分数是分块/按token数加权的文本余弦代理，非独立语义评判，空接收记0。WER可超过100%，不能当1−准确率。

## 阶段与停止条件

1. D0压缩前干净特征（离线oracle，不可部署）、D1冻结B4 Ctext16+T32干净记忆：相同AR初始化，各32训练文本/800更新，实际生成后比较匹配/全零/错配记忆。诊断头不替代正式头。
2. 新网络32文本/800更新过拟合：覆盖WER<=10%、零/错配记忆WER至少恶化10个百分点；失败为`overfit_stop`，不扩大训练、不降低门槛。
3. **主训练从初始权重重新开始，不沿用32句诊断权重**。2000训练文本/8轮/batch4=4000更新。64独立验证文本要求覆盖WER<=20%、记忆干预至少恶化10个百分点；失败为`clean_validation_stop`，不进入FSO。20%是中间工程门禁，不是优质文字最终标准；后续应进一步降低WER并做冻结后的test。现有validation曾被历史使用，不声称全新盲测。
4. 通过后，对真正384复符号短帧重新公开波形校准，train16+holdout8帧/条件。不能截取旧2304载荷bank。当前T0/T1/T2载荷须通过独立噪声全波形holdout的公开误差容差；失败停止，不放宽阈值。
5. T1继续干净训练、T2 FSO带噪训练，从相同T0开始，各2000文本/2轮/batch4=1000更新，相同规模/顺序/LR。T2含25%干净锚，条件内相关天气窗口、每样本独立残差噪声。代理是五tap广义线性算子+相关随机回放，不是可微硬件。失锁训练任务损失0，验证失败仍保留。
6. 最终L0冻结B4文字侧、T0干净、T1继续干净、T2 FSO：64文本×(clean+6条件)×4组=1792行，完整波形单次发送。同路径/噪声身份，实际时长/能量各自报告，不声称不同载荷的受损波形或能量完全相同。L0与新网络是整体架构参考，参数不同，非单一模块因果消融。

主比较T2−T1低质量覆盖WER（负值更好）、图片ID聚类95%区间；一次训练/固定天气，探索性validation，不覆盖总体训练/天气变异。不扫训练种子、不运行最终test、不增加数字基线、不自动接回图文融合。

## 真正文字短帧

48×16=768实数＝384复载荷，R=1，无I224/图像侧C16，无原文UTF8旁路。L0提取分离B4的Ctext16+T32；新网络48槽全为文字，不宣称共享图文Common。

保留64GSa/s、4SPS、16GBaud、RRC0.1、IQ-MZM/EDFA、湍流/指向/功率与V3同步接收。重新计费probe/前导/插入导频/脉冲尾/guard，不把时长算成旧帧六分之一。保存功率、实际模拟SNR、导频SNR、EVM、时长、能量、星座图。连续特征BER为null，不捏造误码率。短帧降低串行化时长，不等于改变频谱带宽；光排程不含神经计算时间。

## 上传解压

把`v10_text_jscc_v1_fix1.tar.gz`和`.sha256`上传到现有`/data1/zw/WYH/multimodel/`，使用当前上传工具，或在本地包所在目录：

```bash
scp v10_text_jscc_v1_fix1.tar.gz v10_text_jscc_v1_fix1.tar.gz.sha256 zw@你的服务器地址:/data1/zw/WYH/multimodel/
```

服务器执行：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_text_jscc_v1_fix1.tar.gz.sha256
test ! -e _v10_text_jscc_v1_fix1 && tar -xzf v10_text_jscc_v1_fix1.tar.gz
"$PYTHON_BIN" -B _v10_text_jscc_v1_fix1/install.py --project-root .
"$PYTHON_BIN" -B v10_extension/text_jscc_v1_fix1/install.py --project-root . --verify-only
```

安装器预检查旧依赖/新源码，冲突不覆盖，HANDOFF原字节备份再追加。此前joint_jscc_arch_v1应已安装并完成；residual_fusion_v2不是必要依赖。已有解压目录先核验，不覆盖修改过的源码。

## GPU1启动

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used,memory.free --format=csv
(
  set -C
  nohup "$PYTHON_BIN" -B -u v10_extension/text_jscc_v1_fix1/launch.py pipeline \
    experiment_results/v10_text_jscc_v1_fix1_01 --max-hours 4 \
    > experiment_results/v10_text_jscc_v1_fix1_01.launch.log 2>&1 < /dev/null &
  echo $!
)
```

CPU/必须Torch合同 -> 真实8训练/2验证文本短预检 -> 正式诊断/训练。smoke绕过质量门禁仅验证接口，明确标记，非正式通过。启动空闲显存不足16GiB停止/等待，运行低于4GiB reserve停止，不操作别人进程。无优化器/RNG精确续训、不自动重启。已有输出改`_02`且同时改日志名，不删旧证据。

默认权重依赖（保留服务器）：

```text
experiment_results/v10_joint_jscc_arch_v1_01/v10_screen/train_B4_separate/fso_delta.pth
experiment_results/v10_fso_fluctuating_v2_01/v10_screen/train_B2_fso_ft/last_delta.pth
experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth
experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth
```

实际目录不同，追加`--source-run experiment_results/实际B4目录/v10_screen`、`--legacy-run experiment_results/实际V2目录/v10_screen`。只有轻量记录、没有权重的解压包不能作依赖。

## 时间、进度、发回

按预检5–15分钟、正式筛选1–3小时规划，**新文字训练时间尚未实测**；800步诊断、实际逐词生成和GPU竞争决定耗时。预检45分钟/整体4小时是停止预算，不是保证时间。训练每100更新打印测量stageETA。

```bash
tail -n 40 experiment_results/v10_text_jscc_v1_fix1_01.launch.log
tail -n 40 experiment_results/v10_text_jscc_v1_fix1_01/v10_smoke.log
tail -n 40 experiment_results/v10_text_jscc_v1_fix1_01/v10_screen.log
pgrep -af 'text_jscc_v1_fix1/(launch|run).py'
```

完成或质量门禁停止，自动收集`experiment_results/v10_text_jscc_v1_fix1_01/v10_text_jscc_v1_fix1_records.tar.gz`。发这个包：全部恢复句子、诊断/训练/门禁、指标、光波形资源/星座图、小型公开短帧bank。大权重留服务器。异常无包则发外层日志、最新screen/smoke日志和FAILED.json。报告`v10_screen/report.html`，汇总`summary.json`。

## 文献与边界

[DeepSC](https://arxiv.org/html/2006.10685v3)：借鉴文本/信道编解码联合训练；Europarl、AWGN/瑞利、理想CSI与本项目不同，不复现MI目标/论文曲线，也不保证连续优于数字。[CLIP](https://arxiv.org/abs/2103.00020)：图文对齐预训练不保证逐字可逆；本版保留作初始化/参考，不把它认定为已证实的唯一瓶颈。

本地无Torch，真实模型加载、梯度/AR生成、显存和改善幅度必须服务器验证；语法/NumPy波形/部署检查不能证明模型有效。
