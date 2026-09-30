# V2：比特级相干 FSO 波形链路（不使用 KK）

状态：本地 NumPy 数值测试已执行；服务器 PyTorch 适配器、真实 COCO/权重、GPU、实际光链路未执行。旧版 `link_compare/`、V10 主模型、划分、权重和结果均不改写。本模块只做 validation 的零微调信道诊断，不用于训练。

## 参照与取舍

依据本项目对话“解读实验室代码”（2026-09-30）及其 `Tx_ex.m`，采纳：Gray 16-QAM（I/Q 两路 PAM4）、14 GBaud、64 GSa/s、RRC 滚降 0.01、8 位 DAC。该对话用户明确不使用 KK，故不移植频移载波、CSPR、Hilbert/log KK 重建。

不是 MATLAB 代码逐行复刻：原实验 PRBS 换成真正的语义载荷比特；不用 I/Q 各自归一化；不用整份源数据训练接收器。直接在 AWG 时间栅格合成分数相位 RRC，14 GBaud 对应精确 32/7 samples/symbol；不依赖 scipy/MATLAB。实验室原 RRC 跨度16，本版默认128以减少极低滚降时的截断ISI，守护样本全部计费；`--span 16` 可单独诊断，不推荐默认冒充等价。

本版接收机：公开256符号前导序列、样点级帧起点搜索（0~32）、RRC匹配滤波、导频复增益/相位估计、单抽头均衡、Gray软解映射。符号时钟已知，块内相位恒定。未实现实验室 CMA/DD-LMS/Volterra、载频偏差、激光相位漂移、采样时钟偏差或双偏振；不是可直接接真实示波器的完整DSP接收机。

## 实际数字链路

```text
原 V10 图像/文字编码与 C/I/T 打包（不变）
 → 固定范围[-4,4]的4/6/8位实数量化（默认8位）
 → SC02/长度/位深协议头 + 量化比特 + CRC32
 → 公开加扰 → 无FEC或K7卷积码 → Gray QPSK/16-QAM
 → RRC → 64GSa/s采样 → 固定满量程8位DAC
 → 相干光场 h=sqrt(湍流×指向×路径强度增益)×exp(jφ)
 → 复热噪声 → 前导同步/匹配滤波/导频均衡
 → 软解调 → 可选Viterbi → 解扰/CRC → 实数反量化
 → 原 V10 图像和文字接收解码器（不变）
```

`16qam_none` 为真实比特流无FEC；`16qam_conv` 使用1/2码率、约束长度7、八进制生成多项式171/133，MSB为新输入，每1024信息比特补零并加6个终止零，软Viterbi回到零状态。**不是LDPC**；原实验室源码未提供数据FEC，这项为新增可关闭对照。CRC失败整帧擦除，不丢弃失败样本、不回退发送端数据、不ARQ。源数据仅用于编码与事后BER/MSE诊断，接收函数不接收源数据。

`analog` 为连续语义复符号经过同样的RRC/DAC/相干链路，无源比特/FEC；保持V10公开token功率UEP。数字分支使用固定码率、均匀保护，不做模拟token幅度UEP。**因此不能把数字/连续差值当作UEP消融或严格等时长对照。** 若要研究数字UEP，需要另定义不同分组码率与公开排布，不在本轮偷偷加入。

## 速率、功率和公平性

完整帧4608实数，8位量化为36864载荷比特；加协议/CRC后36976比特。无FEC16-QAM为9244载荷符号；上述卷积编码后76220比特、19055载荷符号。不能沿用旧版2304连续符号来计算数字载荷时间。

探测导频、模式头、载荷分别有256符号前导序列。默认前向开销32探测+64模式头+768前导=864符号，还包含每段RRC尾部样本。模式头与探测也实际走波形链路。反向模式反馈保留原RF控制协议，默认假定1 Mreal-symbol/s，64符号耗时64微秒，单列并计入顺序链路goodput；不冒充光链路开销。

- 16-QAM@14GBaud原始标号速率56Gb/s；FEC净速率还需扣除终止、填充、头部、前导、守护与CRC失败。
- 64GSa/s固定，8/14/16GBaud分别8、32/7、4采样点/符号。采样率和比特率不同。
- 速率试验采用固定平均功率/白噪声PSD假设，参考Es/N0在14GBaud定义；实际Es/N0=SNRref−10log10(Rs/14G)。波形乘sqrt(Fs/Rs)保持不同速率平均样点功率约一致，样点噪声方差=Nref×Fs/14G保持一致；不靠改变FSO增益拟合结果。
- 同图/档位/参考SNR/重复在所有速率与分支共用衰落/相位身份，汇总强制核对；不同长度波形不能声称噪声样点逐一对应。
- `--analog-bandwidth 10000000000` 可加入假定的10GHz联合前端低通（四阶幅频、零相位），默认不加。未测硬件参数，不自动把64G采样率当作模拟带宽。
- 记录实际DAC clipping、源量化clipping、符号能量、波形平均/峰值功率、能量×时间、前向样本数/秒、FEC前后BER、CRC通过率、语义量化比特goodput。功率是归一化DSP单位，不是已标定光瓦特。不同分支帧时长/能量不相等；先看质量—开销权衡，再谈等时延/带宽预算对照。
- 图像PSNR/SSIM、文本BLEU2及恢复描述沿用原评估器。不执行扩散，不能宣称生成/混合扩散候选已经比较。

## 运行

既有 `v10_extension/link_compare/` 必须存在（V1已部署）。不安装或升级任何服务器依赖，使用既有Python3.8/PyTorch1.12/NumPy>=1.17。

先校验本发布包，解压到项目内暂存目录，再运行附带安装器。安装器只新增 `physical_link/`，遇到已有目标即停，不覆盖旧代码；共用交接文档只添加本版入口并备份旧文档。详细命令见最终交付或 `deploy_and_run.sh`。

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/physical_link/test_core.py
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/physical_link/test_adapter.py
```

两项均通过后再启动 `bash v10_extension/physical_link/run_pilot.sh`。GPU1、8张validation图、0/10/20dB参考SNR、8/14GBaud、analog/无FEC16QAM/卷积码16QAM，共6次评估、144条样本，FP32，不重新训练。输出自动使用新的时间戳目录；预算每评估30分钟、总2小时，是停止预算不是耗时预测。

手动增加QPSK可在同样的新运行中用 `--variants analog 16qam_none 16qam_conv qpsk_conv`。不随意减少SNR或删掉失败图来挑好看的结果。可以先 `--dry-run` 看计划；NumPy仍需存在。

完成后给回 `physical_summary.json`，各子目录 `samples.jsonl`、`summary.json`、`COMPLETE.json`、`run_identity.json`、`checkpoint_load.json` 和根 `design.json`/`COMPLETE.json`，另附保存的图像。这里没有预设数字链路一定胜过连续链路。

## 一手方法核对

[MathWorks RRC设计](https://www.mathworks.com/help/signal/ref/rcosdesign.html)说明单位能量、跨度与滚降，以及有限跨度匹配后的ISI残余；[卷积码trellis](https://www.mathworks.com/help/comm/ref/poly2trellis.html)说明K7、171/133的1/2前馈码。它们支持方法定义，不是本项目性能证据。
