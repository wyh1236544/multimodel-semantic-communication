# FSO JSCC V1：单次连续特征带噪训练

状态：新增候选实验代码；本地 NumPy/协议/安装检查通过。**本地缺少 PyTorch，Torch 合同、真实 A 权重、COCO 与 GPU 训练尚未执行。** 服务器必须先过 check 和 smoke。不替换旧最优版本，不预设结果会提高。

## 这次要回答什么

在相同网络、相同训练预算、相同输入和单次传输资源下，FSO-aware 微调是否比 AWGN 微调更适合低信噪比的连续特征？

|组别|初始化|本轮训练信道|目的|
|---|---|---|---|
|M0_A_frozen|原 V10 base + 精确 A delta|不训练|保留旧 A 的连续传输参考|
|M1_clean_ft|同 A；同新增零残差映射|干净信道|控制额外训练本身的收益|
|M2_awgn_ft|同上|AWGN，名义 −5～20dB|噪声训练基线|
|M3_fso_ft|同上|弱/中/强 FSO，名义 −5～20dB|检验湍流/指向感知训练|

M1 是“追加干净微调”，不是从来没带噪训练过；原 A 已有历史训练。
旧 V10 已存在编码器/解码器联合微调；本轮重新优化当前单次相干连续链路，不能宣传旧训练完全没有联合训练。

默认先做方向筛选：每个新训练组 2000 图 × 2 epochs，batch1、累积2，**每组2000 optimizer steps**。
前100 optimizer steps冻结联合编码器，随后解冻；三组暖启动、步数、样本顺序、初始参数、任务损失完全一致。
不做训练种子扫参，只用 seed42 复现；验证 draw 为独立信道试验，**每次只传一次，不合并恢复**。
32张 validation；6个名义SNR（−10/−5/0/5/10/20）× AWGN/FSO三档 + 干净上界。
−10dB 在训练范围之外。可在方向明确后另开128张验证、4epochs；本轮不自动扩大。

## 模型与信道

保持图像128×128、冻结 VAE/CLIP、CLIP77文本容量与 C32/I224/T32×16：4608实特征 → 2304复载荷符号。
联合语义编码器、瓶颈、联合解码器、图像/文本头、已有接收适配器与两小映射参与微调；其余旧模块冻结。
新增TX/RX残差映射分别为16→32→16、17→32→16，最后一层零初始化；RX额外输入导频可靠性。
TX不输入真SNR/真衰落/接收端可靠性。公开固定排布，无免费原文或内容相关索引。
所有组（含M0）统一FP32，显式关闭原配置的VAE FP16；M0是同精度协议内的原A参考，不是逐位复现历史FP16分数。
图像文本仍通过同一个 C/I/T 联合编码和接收网络；本轮未重新打开历史无明显收益的显式文字引导模块。

训练代理：`h=sqrt(path*GammaGamma_intensity*pointing)*exp(j*phase)`，整帧一个慢衰落系数。
每帧256导频估计CSI；LMMSE型线性均衡，随后可学习修正。
发端按C/I/T组归一化；衰落后不重归一化，也不重新调噪声使实际SNR恒定。
FSO参数在config.json：alpha/beta/xi/A0/path loss。弱/中/强不是实测天气等级。
50%名义SNR在−5～5、30%在5～10、20%在10～20；三种FSO档均匀采样。

任务损失：1.5 latent MSE + .75像素 MSE + .35(1−SSIM) + .5掩码文本CE。
相同RRC波形PAPR惩罚用于所有微调组，不单独给FSO组额外语义损失。
VAE冻结参数但保留解码梯度，不使用伪造的数字CRC/QAM直通梯度。
不加扩散；优先检查原图/原文忠实度。

代理信道不是完整光器件仿真：它忽略 DAC/MZM/ASE/同步失锁。
因此另在**现有 V3 详细波形链路**对固定前8张验证图、4条件、单次发送进行独立复测。
64GSa/s AWG，4SPS → 16GBd，RRC滚降0.1、span32，复包络占用带宽约17.6GHz。
纯连续实特征没有天然比特率、字节压缩比或BER；记录符号/样本/时间/能量/特征MSE/PAPR。
V3另保存光功率、ASE/热/散粒等噪声、离线真实参数SNR、导频SNR、失锁、星座/连续散点与接收效果。
四组使用相同slot/峰值包络/载荷符号数；波形形状会改变实际平均光功率和能量，报告实际值，不宣称严格等能量。
保持历史V3的公开工作点，不按各载荷或真接收SNR额外调光功率。

## 上传、解压、安装

把 `v10_fso_jscc_v1.tar.gz` 和 `.sha256` 两个文件上传到**已有** `/data1/zw/WYH/multimodel/`。
可以用目前连接服务器的文件面板/SFTP直接上传。若用Mac本地终端，主机填写已有SSH配置里的别名/地址：

```bash
scp /本地实际路径/v10_fso_jscc_v1.tar.gz /本地实际路径/v10_fso_jscc_v1.tar.gz.sha256 zw@你的SSH主机:/data1/zw/WYH/multimodel/
```

以下在服务器执行，不复制提示符；不需要安装或升级依赖：

```bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_fso_jscc_v1.tar.gz.sha256
test ! -e _v10_fso_jscc_v1 && tar -xzf v10_fso_jscc_v1.tar.gz
/home/zw/anaconda3/envs/torch/bin/python -B _v10_fso_jscc_v1/install.py --project-root .
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/fso_jscc_v1/install.py --project-root . --verify-only
```

如果已解压，第二次`test ! -e`不再解压，这是保护行为，直接继续安装/核验。
安装只新增 `v10_extension/fso_jscc_v1/`，检查已有依赖与冲突；不覆盖旧源码、权重、划分、结果。
仅共享HANDOFF会在逐字备份后追加新条目。任一依赖哈希不同则安装前停止，请发具体报错，不强行覆盖。

## 推荐：一次启动 check → smoke → 三组微调 → 配对验证 → 打包

先确认你有权使用GPU1，不停止其他人的任务：

```bash
cd /data1/zw/WYH/multimodel
nvidia-smi -i 1
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/fso_jscc_v1/launch.py pipeline experiment_results/v10_fso_jscc_v1_01 --max-hours 12
```

此命令使用物理GPU1，程序内cuda:0；始终FP32，不自动安装环境。
默认原权重：`experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth`。
默认A增量：`experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth`。
如果真实路径不同，增加 `--checkpoint 实际路径 --a-delta 实际路径`，不能混用不匹配的base/delta/划分。

想先只运行短预检：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/fso_jscc_v1/launch.py check
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/fso_jscc_v1/launch.py smoke experiment_results/v10_fso_jscc_v1_smoke01 --max-hours 0.5
```

check里Torch测试是真正执行，不跳过缺失依赖。smoke每组8图×1轮、2验证图，全链路和真实权重通过后才跑quick/pipeline。
如果输出目录已经存在，改新编号；不删除旧证据，也不自动恢复或拼接失败结果。

## 后台运行与进度

下面同样会运行完整pipeline，**与前台命令二选一，不重复启动**：

```bash
cd /data1/zw/WYH/multimodel
test ! -e experiment_results/v10_fso_jscc_v1_01 && test ! -e experiment_results/v10_fso_jscc_v1_01_launcher.log && nohup /home/zw/anaconda3/envs/torch/bin/python -B -u v10_extension/fso_jscc_v1/launch.py pipeline experiment_results/v10_fso_jscc_v1_01 --max-hours 12 > experiment_results/v10_fso_jscc_v1_01_launcher.log 2>&1 < /dev/null &
```

观察：

```bash
tail -n 40 experiment_results/v10_fso_jscc_v1_01_launcher.log
tail -n 40 experiment_results/v10_fso_jscc_v1_01/v10_smoke.log
tail -n 40 experiment_results/v10_fso_jscc_v1_01/v10_screen.log
find experiment_results/v10_fso_jscc_v1_01 -maxdepth 3 -name COMPLETE.json -print
```

screen.log在真实smoke结束后才出现；加载模型/缓存VAE阶段没有TRAIN行是正常的。
训练行区分forward与optimizer step；前100/200 optimizer step保存时间估计，**12h是总停止预算，不是预计耗时**。
每组训练最多3h，超时保留失败；有限值异常不跳过更新。并行使用中的GPU可能显著改变耗时。

## 发回什么 / 能得出什么

发 `experiment_results/v10_fso_jscc_v1_01/v10_fso_jscc_v1_records.tar.gz`，无需大权重、COCO数据。
如果失败，发launcher.log、对应smoke/screen.log和FAILED.json；不要反复重启直到偶然成功。
单独收集已完成screen：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/fso_jscc_v1/collect.py experiment_results/v10_fso_jscc_v1_01/v10_screen experiment_results/v10_fso_jscc_v1_return01.tar.gz
```

HTML显示同图四组恢复、原文与接收文字；SVG显示PSNR/WER曲线和图片间SD；CSV/JSON给配对增益及图像bootstrap CI。
Common/接收文本latent对原文latent余弦同属CLIP家族，只是辅助语义对齐指标，非独立语义证据。
保存整个验证矩阵、低SNR和失锁样本，不能只展示好看的20dB图。
主要看M3相对M2在FSO名义−5/0/5dB的增益；再看高SNR退化、文字忠实度和独立波形链路能否保留收益。
结果不单独证明图文联合优于分开模型、连续优于数字或JPEG。那些需要额外匹配资源的独立对照。

## 文献依据（不是本项目实测结果）

- [DeepJSCC](https://arxiv.org/abs/1809.01733)：可微信道内的端到端训练与慢衰落。
- [ADJSCC](https://arxiv.org/abs/2012.00533)：跨SNR训练思路；本版TX不获得免费真实SNR。
- [DTAT-FSO作者预印本](https://arxiv.org/abs/2409.11928)：噪声训练/PAPR与光通信；未复现论文架构。
- [TRFSO作者预印本](https://arxiv.org/abs/2509.03835)：FSO特定训练及物理链路失配的重要性。本文代理没有实测数据拟合。
- [Farid & Hranilovic](https://doi.org/10.1109/JLT.2007.899174)：慢湍流、路径与指向误差；本版相干场须对光强衰落开平方。

更完整的下一阶段规划：`v10_extension/docs/FSO_NOISE_LITERATURE_PLAN_20261011.md`。
