# 相干 FSO 真实性与连续／数字对照 V2

状态：可运行候选实现，2026-10-05。独立目录，不修改旧链路或 A 权重。默认不训练、不扫描训练种子、不使用 test。先做有界链路检查和小样本方向验证，不能把这版称为硬件标定或 OptiSystem 完整复刻。

## 安装：先上传源代码压缩包

把 `v10_optical_realism_v2.tar.gz` 及其 `.sha256` 上传至现有 `/data1/zw/WYH/multimodel`。不要解压覆盖旧版目录。以下每步成功后才进行下一步：

```bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_optical_realism_v2.tar.gz.sha256
test ! -e _v10_optical_realism_v2 && tar -xzf v10_optical_realism_v2.tar.gz
/home/zw/anaconda3/envs/torch/bin/python -B _v10_optical_realism_v2/install.py --project-root .
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/optical_realism_v2/install.py --project-root . --verify-only
```

解压产生安装暂存目录 `_v10_optical_realism_v2`。已存在时不重新解压，不删除旧目录，先核验其来源。安装器先检查全部依赖哈希/冲突，再新增 `v10_extension/optical_realism_v2`。共享交接文件先按字节备份再追加；其他旧源码、权重、结果不覆盖。需要前一版 `final_compare_v1` 及其原有依赖；版本不符会停止，不能跳过哈希检查。

## 推荐第一次运行：GPU1，不重训

```bash
cd /data1/zw/WYH/multimodel
CUDA_VISIBLE_DEVICES=1 /home/zw/anaconda3/envs/torch/bin/python -B -u v10_extension/optical_realism_v2/launch.py pipeline experiment_results/v10_optical_realism_v2_quick_01 --max-hours 4
```

新流水线依次执行：

1. NumPy 数值/协议/验收测试，及已有 PyTorch `ReplayFrame` 接口测试。
2. 15 个背靠背波形诊断：QPSK、16QAM、连续高斯载荷 × 理想附加损伤关闭／仅载波／仅时钟／仅器件／全部附加损伤。保留原有光放大噪声和带宽，`ideal` 不等于无噪声。固定30dB衰减器，不含自由空间随机衰落。
3. 2 张 validation 图的真实 A 冒烟检查，3 工况、1 次通道抽样。
4. 两份 8 张 validation 对照：QPSK+卷积码，以及16QAM+卷积码+2次比特重复。每份包含 JPEG+原文、语义数字、语义连续，以及三项无信道参考。共每份96行，两个策略192行；另有24行冒烟。工况为 good/-9、moderate/-24、stress/-9 dBm公共校准参考点。
5. 自动验收、图片/文字/星座报告和结果打包。

这些是通道抽样，不是新增训练种子。无神经网络训练。`4小时` 是整个流水线停止上限，不是预测耗时。新增 DSP 尚无你的服务器计时；先看 smoke 实测用时及各目录 `progress.json` 的预计总时间。若失败保留现场，不自动扩大预算或跳过检查。

固定使用既有路径：

- 基础权重：`experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth`
- A 增量：`experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth`
- COCO、VAE、CLIP、固定划分沿用上一版。载入会验证基础权重/A增量/划分身份。

没有启动任何服务器任务。上述命令由你执行；程序不会停止其他 GPU 用户的进程。

## 进度与交回结果

```bash
tail -n 15 experiment_results/v10_optical_realism_v2_quick_01/smoke.log
tail -n 15 experiment_results/v10_optical_realism_v2_quick_01/qpsk_conv.log
tail -n 15 experiment_results/v10_optical_realism_v2_quick_01/16qam_repeat2.log
```

后面阶段尚未开始时，相应日志不存在是正常的。完成后发回一个文件：

`experiment_results/v10_optical_realism_v2_quick_01/v10_optical_realism_v2_records.tar.gz`

如果失败，发对应日志最后80行，不需要传权重/数据集。输出已存在时换 `_02`，不要覆盖或清理失败现场。

每个策略目录内 `report.html` 展示同图原图/恢复图、完整原文/恢复文本、星座；`summary.json` 有图像聚类配对统计；`samples.jsonl` 和 `packets.jsonl` 保留真实功率、导频估计、离线噪声分项SNR、EVM/SNDR、BER、锁定判断、时钟/CFO估计、样点、光能量、峰均比和削顶。全部失败参与计分。连续 BER/CRC 不适用，不能写成0。

## 这次具体改变了什么

| 部分 | V2实现 | 默认假设 |
|---|---|---|
| 载波 | 发射/本振 Wiener 相位噪声、LO频偏，接收导频估计与插值跟踪 | 两端各100kHz线宽，LO偏移+50MHz |
| 时钟 | ADC量化前的分数采样、固定ppm偏差、逐样点抖动；接收端独立搜索和拟合 | +20ppm、0.35样点、0.2ps RMS |
| 器件 | 理想DAC/ADC之外的ENOB附加白噪声；I/Q增益/时延不平衡；MZM正交误差 | DAC6/ADC10有效位、0.5dB、0.15样点、3° |
| 导频 | 256前导；每256个载荷符号后加32个跟踪符号；最后一段也付开销 | 两条路线共同格式，无连续载荷QAM判决 |
| 功率 | 公共1024符号QPSK校准后固定激光/MZM/EDFA工作点 | 不读取当前业务波形或接收真值逐包调功率 |
| 资源 | 实际RRC样点、导频、包头、CRC、FEC、重复、尾部和保护间隔 | 64GSa/s、4SPS、16GBd、滚降0.1、4µs上限 |
| 失败 | 导频相关/残差/相位歧义/时钟捕获边界触发拒收，载荷清零并保留计分 | 门限为公开假设，尚未实测校准 |

注意频偏符号：配置是 LO 相对 TX 的偏移；复接收信号的 CFO 约为负50MHz，并非符号算错。短前导不足以稳健估计ppm时，报告0代表未从长时跨度拟合，不是知道时钟完全同步。

同样4µs、加入跟踪导频后，连续最多重复24次（旧版27次不再可行）。QPSK卷积码与16QAM卷积码+重复2的有效载荷编码符号成本相同，但星座距离不同。两者都是固定策略对照，不是自适应选择，也不是没有FEC的弱基线。

所有路线固定相同激光设置及峰值包络/时间上限，**并不保证实际平均光功率或消耗能量相等**。`launch_target_signal_dbm` 为兼容字段，现指公共校准参考点；实际发射/接收功率必须看 `power_stages`。连续优势应由质量—实际时间/能量曲线及失败率证明，不能只比较重复后的 PSNR。

## 配置与后续可选运行

基础光学参数仍在 `optical_link_v1/optical_config.json`；新版附加参数在 `hardware.json`，路线/公共时间预算在 `plan.json`。建议复制成新的项目内配置，使用 `--hardware-config`、`--optical-config`、`--plan` 指定，保留发布源码。数值带单位，严格拒绝未知键/非法范围。所有未测设备值标记为假设；用户提供64GSa/s不代表其他参数已标定。

只验证波形（无需权重/GPU）：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/optical_realism_v2/launch.py b2b experiment_results/v10_realism_b2b_01
```

关闭本版附加损伤作对照，仍使用新版计费导频和固定工作点；不是复写旧实验：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B -u v10_extension/optical_realism_v2/launch.py quick experiment_results/v10_realism_ideal_01 --preset ideal
```

研究连续重复的实际收益，可用新输出目录分别给 `--analog-repeats 1`、`3`、`9`；默认 `max_fit`=24。不要先全部运行，先看主流水线是否有效。27在当前4µs下会明确报错；增加公共slot预算后才允许，不能偷偷减少导频。

确认方向后才运行 `validation`：两策略各32张×9工况×3通道抽样，仍无训练，并且受最大6小时整体上限约束。默认流水线不自动进入这一大规模阶段。当前没有新的最终test命令。

## 尚未做到、不能夸大的部分

- 保留的物理链路已含Gamma-Gamma、指向/孔径/耦合损耗、EDFA/ASE、相干探测、散粒/热噪声；本版仍是一业务内常量块衰落，未增加空间相位屏、实测耦合时间序列、跨业务相关湍流或完整捕获/重捕获状态机。
- ENOB只是白噪声替代，未加载实测复频响；脉冲/滤波仍是离散包络模型，不是光载波逐周期仿真。接收定时为有限搜索+缓冲导频拟合，不是已标定Gardner/Farrow实时实现。
- 离线物理SNR分母是光学/散粒/热噪声；ENOB/量化、削顶、相位/时钟残差等由EVM/SNDR反映。导频残差有估计偏差，两种SNR都不是仪器实测真值。
- 相位插值与重复合并的残差可能相关，`variance/R`可靠性只是近似；实际恢复MSE及业务指标照常统计。
- 数字仍是K7卷积码/整对象CRC，尚未加入LDPC/Polar、语义分块恢复、分层JPEG或混合残差。新增QPSK对照不等于已优化所有数字基线。
- 文字在无信道A模型中的损失不会因DSP修复消失；语义路线仍只传联合特征，不偷加原文旁路。
- 未做实际GPU/COCO推理或服务器测试，未证明PSNR增益；本地数值通过见 [VERIFICATION.md](VERIFICATION.md)。

方法依据：上一轮调研采用 OptiSystem 相干DSP、OptiCommPy 同步/载波处理文档、NI转换器指标，以及 DeepJSCC/DeepJSCC-Q 文献。这里为无新增依赖的透明NumPy候选实现，非复制厂家模型。
