# V7：冻结发送端的双向接收辅助验证

这是独立、可回退的接收端实验，不替换任何 V6/旧版本。

## 要回答的问题

在同一份收到的 C/I/T 波形、同样载荷和尽量匹配网络规模下，文字记忆是否能帮助恢复图片，图像特征是否能帮助恢复文字？

- **B0_frozen**：现场计算的冻结参考。V6 J0 的 C/I/接收桥接权重 + V6 独立文本 codec 的全局 9000 步权重。不冒充历史 V6 的原样重放。
- **R0_self**：新增相同大小的接收适配器，只允许各自模态内部注意力。
- **R1_text_to_image**：允许收到的文字记忆帮助图片；文字不看图片。
- **R2_image_to_text**：允许收到的图像特征帮助文字；图片不看文字。
- **R3_bidirectional**：同时开放两个方向。两个方向都使用冻结基线的特征，不循环喂回生成结果。

每组相同注册参数、相同初始化、相同训练顺序/步数、相同损失和同一份噪声。注意：有效计算量/有梯度参数可能不同，不能声称 FLOPs 完全相同。适配器 hidden128、两层注意力、零初始化残差输出；第一步输出必须与 B0 一致。全部旧参数、BART、CLIP、VAE 保持冻结。仍保留 C/I/T，不重新训练 Common 或发送端，不使用扩散，不传原始 UTF8。

四组均额外计算冻结 CLIP 对“接收端初步重建图片”的语义特征，以避免小适配器完全从头学习视觉语义。该特征只在接收端产生，不读取原图或发送端 CLIP 特征，也不额外占信道资源。R0/R1 的文字支路禁止访问它；R2/R3 可通过交叉注意力访问。所有组都有同样的视觉语义投影层。默认 BART768/CLIP512 时新增可训练参数约0.83M；冻结模型本身仍占显存/计算。

3072 个复符号，R1，C512/I3584/T2048 个实数。64GSa/s、4SPS、16GBaud、RRC0.1；帧头、导频、保护间隔等通过旧物理链路的资源函数全部计入。连续特征没有定义比特映射，因此 BER 为 null，不编造比特率/压缩比收益。

## 训练与验证

clean 与 fso 从完全相同的未训练适配器开始。fso 的 75% 前向使用已有 Gamma-Gamma/指向误差/加性噪声/导频估计的符号级近似，25% 是 clean anchor。代理没有训练完整的 DAC/MZM、同步失锁、天气时间相关性，不能叫硬件数字孪生。两阶段均使用固定轮换的 25% 私有特征丢弃，明确是人工诊断，不是物理包丢失。

验证每张图在每个条件只生成一次完整物理接收波形，供所有解码器和对照复用。真实 FSO 包含弱/中/强条件以及接收失锁；失败计入平均值，输出黑图和空文本。额外 `clean_drop_image/text` 是单独标记的人工诊断。clean 输入的 40dB 只是有限条件哨兵，不是测得 SNR。

R3 另有 cross_off、零文字辅助、零图片辅助、打乱文字辅助、打乱图片辅助；只改“辅助视图”，不改被恢复对象自身收到的特征。打乱供体也是同一条件真实收到的其他样本，不使用供体干净原文/原图。整帧失锁时，任何辅助都不能越过可用性门。

loss=文本 CE + latent MSE + 0.1像素 MSE + 0.01冻结 clean 接收记忆 anchor。训练目标可使用 TRAIN 原文/原图；接收接口不接收原文、原图、原始长度、原始 mask。固定最后权重，无验证选优；无精确续训声明，无失败跳步。

## 安装与第一轮运行（GPU1）

把源码压缩包上传到 `/data1/zw/WYH/multimodel`。在服务器执行：

```bash
cd /data1/zw/WYH/multimodel
sha256sum v10_bidirectional_rx_v7.tar.gz
test ! -e _v10_bidirectional_rx_v7 && tar -xzf v10_bidirectional_rx_v7.tar.gz
/home/zw/anaconda3/envs/torch/bin/python -B _v10_bidirectional_rx_v7/install.py --project-root /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/bidirectional_rx_v7/launch.py diagnostic experiment_results/v10_bidirectional_rx_v7_01 --max-hours 3
```

默认 parent 是 `experiment_results/v10_clip_bart_joint_v6_03/v10_screen`；若真实完成结果目录不同，请显式 `--parent-screen /绝对路径/v10_screen`，不要混入 smoke 权重。服务器必须保留 V6 clean/last.pth、text_prefit/clean/last.pth 和已有 B4/B2/A/基础权重；只上传 records 包不能恢复这些权重。

`check` 仅源码、CPU 合约检查；`data-check` 只核查原始数据可用性。所有训练动作先运行实际 smoke（TRAIN8/VAL2），然后新建主目录：

| 动作 | 每组 TRAIN/VAL/轮数 | 训练组数 | 用途 |
|---|---|---|---|
| smoke | 8/2/1 | 4种接收器×clean/FSO=8 | 真实模型接口、梯度、磁盘重载、安全门，不是效果结论 |
| diagnostic | 32/8/8 | 8 | 先确认机制能不能学习；256前向/组，128更新/组 |
| screen | 256/32/4 | 4，只有FSO训练 | 低成本扩大验证；不能据此单独归因噪声训练收益 |
| pipeline | 256/32/4 | 8，clean与FSO | 配对区分融合收益和带噪训练收益 |

**先只跑 diagnostic，把结果发回来再决定扩大。** 新动作不加载 smoke/diagnostic 训练权重；总时间上限不是预计完成时间。初步未实测估计：smoke5–15分钟；diagnostic含评估约30–90分钟；完整pipeline约1–3小时，取决于生成文本长度/GPU共享。`progress.json` 及日志的 ETA 只估计当前训练臂，不包括其余训练和波形/文本验证。

```bash
tail -n 30 experiment_results/v10_bidirectional_rx_v7_01/v10_diagnostic.log
cat experiment_results/v10_bidirectional_rx_v7_01/v10_diagnostic/progress.json
```

遇到已有目录请换 `_02`，不要删除旧目录。启动须至少16GiB空闲显存；运行保留4GiB；不安装依赖，不停止其他用户进程。

## 看结果与收集

完成后发送 `experiment_results/v10_bidirectional_rx_v7_01/v10_bidirectional_rx_v7_records.tar.gz`。

包含 summary.json、REPORT.md、实际图文 HTML、所有评价文本、固定前几张图片、接收星座 SVG、共享接收帧 NPZ、训练历史和梯度/冻结/重载审计；不包含模型权重、完整数据或任何凭据。HTML 可在解压后的本地直接打开。

主比较预先固定为中等真实 FSO 条件下同训练阶段的 R3−R0；PSNR差正、WER差负才是向好。条件均值包括失锁。R1/R2用于定位辅助方向；关掉/置零/错误配对必须核查是否削弱收益。报告只提供条件于一个训练权重的图像簇 bootstrap 区间；8/32张重复验证不是独立测试，不保证显著提高，不宣称发表概率。

如果只在人工丢弃特征时有效、真实FSO时无收益，结论只能是特定互补损伤下的机制有效，不能泛化成 FSO 优越性。只有独立测试、更多训练/数据和复现实验才能形成最终论文证据。当前 C/source 冻结，即使 RX 有收益，也不能直接宣称联合源编码/压缩/节省带宽已实现。
