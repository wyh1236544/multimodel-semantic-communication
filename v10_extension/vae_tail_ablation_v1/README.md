# 图文联合模型：VAE 解码器末端局部解冻消融 V1

状态：新增代码；本地非 Torch 合同/合成报告已验证，真实骨干/GPU 训练待服务器预检。2026-10-12。不是已确认增益，也不是纯图像论文复现。保留现有图文联合结构、旧代码/权重/结果。物理 GPU1，FP32，不升级环境，不做训练种子扫描或最终 test。

## 实验问题与三组

从同一份已完成 `fso_fluctuating_v2` 的 **B2 FSO** 权重初始化：

| 组 | 新训练 | VAE 状态 |
|---|---|---|
| U0_frozen | 无，既有模型参考 | 全冻结 |
| U1_fso_frozen_vae | 现有语义编码器/瓶颈/信道映射/联合解码器/任务头继续 FSO 微调 | 全冻结 |
| U2_fso_vae_tail | 与 U1 同数据、噪声、顺序、损失、优化器步数及语义层学习率 | 仅最后一个 `decoder.up_blocks`、`conv_norm_out`、`conv_out` 可训练 |

CLIP 全部、VAE 编码器/中间块/更早解码层均冻结并逐参数检查未变。训练的是已有权重，不添加新骨干。U2 比 U1 的额外可训练数量由真实模型打印、保存，不将整个约8370万 VAE 参数都解冻。局部解冻学习率1e-6，尾部权重衰减0；语义层沿用 V2 学习率与损失。15%干净锚点替代步骤；所有失败帧保留，相同不可用帧屏蔽任务梯度，PAPR仍优化。冻结 VAE 权重不阻断对收到的 latent 的梯度；本版本保留原解码路径，没有 no_grad/伪导数。

同预算指相同图文数/更新数/链路资源，不是相同可训练参数或训练计算量；增加的算力与显存单独报告。两臂各1000对训练图文×3轮=3000次前向、1500次更新。单个训练身份42，非多种子扫参。最后一步权重，禁止看验证图挑最优。

## 参考恢复与评估

先由 U0 保存图像直接 VAE 重建和无噪声完整模型恢复；训练后对三组进行：

- **直接 VAE 重建**：完全绕过语义压缩和信道，使用同一份缓存 posterior sample，比较恢复损失。
- **相同 U0 语义 latent 输入**：三组解码器接收完全相同的 U0 无噪声语义恢复 latent，查看末端解码器变化的独立输出作用。
- **完整无噪声/FSO 图文恢复**：完整网络各自编码并恢复。记录 PSNR/SSIM、真实接收文本/原文评价参考、WER/CER/BLEU/token准确率/Common相似度、光功率/SNR/EVM/失锁、时间/能量、图像与星座图。

直接 VAE 分数是**参考，不是数学上界**；不将其与完整模型差值武断解释为唯一原因。只更新尾部不能保证解决文本错误，也不能证明冻结全部骨干是历史无增益的根因。

默认验证64对，clean+6档×2独立信道试验×3模型=2496行，另有192行 VAE 诊断。独立信道试验不合并、不重复发送，R=1。预先规定主比较：验证位置32-63的32张，四个原 V2 低质量档位，U2-U1双方可用 PSNR。前32张用于历史诊断，全部64张/分档/干净/固定latent差值作探索性对照。这些验证图可能用于过早期扩展实验，**不是全历史未见或最终 test**。

报告按图片聚合的配对bootstrap区间，同时给全部帧、双方失锁默认输出与不一致可用性贡献。单训练重复、固定天气轨迹，区间不涵盖训练/天气总体不确定性；双方可用是处理后条件，不能单独支持因果有效吞吐结论。预选图片位置0、1、32、33，不挑最好看的。

## 链路和已有校准

复用用户已完成 V2 的校准 `bank.npz/bank.json/COMPLETE.json`；严格检查来源/哈希/配置。**不重新校准、不混用 smoke bank**。相同随机代理进入两臂训练，最终完整 V3 波形评估绕过代理。近似代理的载荷分布失配限制仍存在，不宣称硬件测量或论文1:1复现。

保持64GSa/s、4SPS、16GBaud、RRC.1，C32/I224/T32宽16、2304复载荷符号，R=1，无UTF8文本旁路。湍流/指向/器件/DSP与原V2相同；SNR来自功率和噪声，失锁不删除。相同峰值/时间上限不等于逐帧相同实际平均能量；能量仍记录。无数字基线、不证明连续>数字或联合>分别。

## 上传与安全安装

把 `v10_vae_tail_ablation_v1.tar.gz` 和同名 `.sha256` 上传到已有 SFTP 连接中的 `/data1/zw/WYH/multimodel/`。压缩包只有源码/配置/测试/文档，不含权重数据。随后：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_vae_tail_ablation_v1.tar.gz.sha256
test ! -e _v10_vae_tail_ablation_v1 && tar -xzf v10_vae_tail_ablation_v1.tar.gz
"$PYTHON_BIN" -B _v10_vae_tail_ablation_v1/install.py --project-root .
"$PYTHON_BIN" -B _v10_vae_tail_ablation_v1/install.py --project-root . --verify-only
```

每条成功再执行下一条。源码/依赖冲突就停止并发回错误，不覆盖或删除旧模块。安装器仅添加这个模块，追加交接前保存原文备份。相同版本重复安装可验收，不改旧结果。

## 一条完整实验命令

先检查 GPU1 占用，不停止其他人的进程。已有 V2 的大权重/校准必须留在服务器。

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
nvidia-smi -i 1 --query-gpu=index,name,memory.used,memory.free --format=csv
"$PYTHON_BIN" -B -u v10_extension/vae_tail_ablation_v1/launch.py pipeline \
  experiment_results/v10_vae_tail_ablation_v1_01 --max-hours 3
```

流程：CPU数值/冻结合同→真实已保存权重的8对预检→两臂小规模训练→64对波形评估/诊断→独立验收/打包。预检通过才继续，所有权重状态重新从同一初始化建立。预检不是主实验训练的一部分。

默认初始化：`experiment_results/v10_fso_fluctuating_v2_01/v10_screen/train_B2_fso_ft/last_delta.pth`；校准自动取 `experiment_results/v10_fso_fluctuating_v2_01/calibration`。如果你上一轮用了 `_02` 等别的目录，请显式加：

```bash
"$PYTHON_BIN" -B -u v10_extension/vae_tail_ablation_v1/launch.py pipeline \
  experiment_results/v10_vae_tail_ablation_v1_02 --max-hours 3 \
  --source-run experiment_results/v10_fso_fluctuating_v2_02/v10_screen
```

这里 `_02` 仅是不同目录示例，必须替换为真实已完成的 V2 目录；不是自动重跑。需要单独指定校准时可加 `--calibration 实际校准目录`。

仅CPU检查：`"$PYTHON_BIN" -B v10_extension/vae_tail_ablation_v1/launch.py check`；仅8对预检：`"$PYTHON_BIN" -B -u v10_extension/vae_tail_ablation_v1/launch.py smoke experiment_results/v10_vae_tail_smoke_01 --max-hours 0.5`。入口强制物理GPU1，程序见cuda:0，局部环境不写全局配置。

## 耗时、进度、结果

历史 V2 每臂6000前向约23分钟；本次每臂3000前向，冻结臂约12分钟作初始参考。新尾部反传未实测，两臂训练暂估25–35分钟，加预检/加载/缓存/2496行评估，**先留35–60分钟**。GPU争用/I/O/新尾部反传可能超出；这不是保证。8步预检投影很粗，第100–200步日志修正。3小时是总安全停止上限，2小时是每臂训练上限，不是预计时长。

另开终端查看：

```bash
cd /data1/zw/WYH/multimodel
tail -n 20 experiment_results/v10_vae_tail_ablation_v1_01/v10_smoke.log
tail -n 20 experiment_results/v10_vae_tail_ablation_v1_01/v10_screen.log
```

日志打印真实新增参数数量、训练投影；结果记录GPU型号、每臂峰值 allocated/reserved 显存和训练时间。8GiB空闲/磁盘预检只是启动门槛，不保证正式训练不会OOM；若失败保存日志，不自动减batch或改变实验后重试。

完成后发回：
`experiment_results/v10_vae_tail_ablation_v1_01/v10_vae_tail_ablation_v1_records.tar.gz`。
包含完整统计、图文/星座图、直接VAE/固定latent诊断、训练记录与参数/显存证据；新权重保留服务器，不打入小结果包。失败则发对应日志和FAILED.json。旧成果全部保留。

验收只确认实验正确执行，不保证增益。若 U2 仅改善失锁默认输出、退化干净参考或区间不明确，就保留该负结果，不自动扩大训练或解冻 CLIP。
