# 图文联合连续 FSO：全解冻筛选 V1

状态：新增实现，本地非Torch检查已执行；真实CLIP/VAE/GPU训练待服务器预检。2026-10-12。
不覆盖旧代码、数据、划分、权重或结果；不升级现有环境。默认物理GPU1，FP32。

## 这次比较什么

| 组 | 初始化 | 新训练 | 范围 |
|---|---|---|---|
| U0_frozen | 已完成V2 B2 FSO | 无 | 原始模型参考 |
| U1_fso_vae_tail | 同一B2 | 500对×2轮 | 原有语义/信道/恢复网络＋VAE末端，约2920万参数 |
| U2_fso_full | 同一B2 | 同样500对×2轮 | 所有已注册参数具备训练资格，包括CLIP文本、VAE编码/解码及当前有效辅助层 |

U2约2亿注册参数具备资格，但不会强行启用停用分支，只有进入当前损失计算图的参数得到梯度/优化器状态。
资格数量不等于真正更新数量：输出逐模块梯度、逐参数变化、非零梯度覆盖数量。
原先约1.75亿是CLIP文本＋VAE＋语义主干的范围估计；本版按用户全解冻要求还纳入已有有效辅助层，实际数量以模型记录为准。
不是训练CLIP图像塔，也不新增或训练扩散UNet。

两臂从同一原始B2初始化重新训练，不能拿长训全解冻和历史短训局部解冻直接比。
每臂1000个前向、500个更新；batch1累积2，前25个更新保持原语义编码器暖身规则。
语义编码器5e-6/恢复2e-5/信道映射5e-5；双方VAE尾部1e-6，U2其余CLIP/VAE和辅助层1e-6。
新增骨干/辅助参数不加权重衰减，原语义层沿用1e-5。全部dropout关闭以控制随机性，不等于关闭梯度。
单训练身份42用于复现、配对随机损伤，不做多种子扫描。

## 真正训练编码器，而不是只打开参数开关

训练/完整评估每次调用**当前**VAE编码器、CLIP文本编码器，绝不把缓存latent作为模型输入。
原始B2在训练开始前保存固定latent监督与CLIP评价参考；用于loss/评价，不送入接收端。
每张图固定同一个标准正态posterior epsilon，两臂使用当前encoder的mean/std重参数化，保证变化来自权重而非随机抽样差异。
CLIP的冻结no_grad分支、VAE的detach和逐步梯度开关已按组调整；U2的CLIP骨干/投影、VAE编码器/解码器必须有非零梯度与实际权重变化。
该设计保留原latent监督坐标，避免移动监督目标；它不是去掉全部预训练约束从零训练。

损失双方相同：1.5固定参考latentMSE＋.75像素MSE＋.35(1-SSIM)＋.5文本CE＋5e-4PAPR。
15%干净锚点替代训练步骤，不额外增加训练量。双方相同失锁帧屏蔽任务梯度，仍记录失败并训练PAPR。
只保存固定最后一步，不根据验证图选最优。缩短训练仅作方向筛选，负结果不排除更充分训练。

## 信道、诊断与结论边界

保持V2相关波动FSO、原完整校准bank和V3波形接收机，不重新校准。
64GSa/s、4SPS、16GBaud、RRC0.1；C32/I224/T32、宽16、2304复载荷符号，R=1。
训练使用有相关随机残差的可微近似代理；最终独立波形评估绕过该代理。仍是模拟器拟合，不是硬件测量。
编码后载荷分布变化可能增加代理失配；本小试验未新增校准或证明代理对所有学到的载荷都精确。
没有UTF8文字旁路、免费重复、模拟传输BER，也没有数字/JPEG或图文分别传输新对照。
同样峰值/时隙上限不代表逐帧实际能量严格相同，实际能量继续记录。

验证64对图文：clean＋6条件×2独立链路试验×3组＝2496行，另192行诊断。
两个draw是独立单次试验，不是重复发送合并。验证图可能在历史实验用过，不是全历史未见/最终test。
预定主比较为位置32–63的32张、4个低质量条件、U2-U1 **全部帧**PSNR。
另报告双方可用、双方失锁默认输出、可用性不一致及失锁率，不删除失败帧。
同一组图计算 `(U2噪声PSNR-U1噪声PSNR)-(U2干净PSNR-U1干净PSNR)`：正值提示噪声损失变小，但单次短训不能作普适抗噪证明。
文字WER/CER/BLEU/token准确率与图片PSNR/SSIM同时保留；文本指标按模型77-token截断能力注明限制。
Common相似度使用原始冻结CLIP参考，避免拿各组变化后的目标互比；同族语义指标仍需独立评价补充。
按image_id聚合bootstrap，区间条件于固定天气轨迹、单训练重复，不覆盖天气/训练总体不确定性；分档和多个诊断是探索性，未作多重检验显著性宣称。

诊断包括：同一原始B2 posterior的直接解码、当前VAE自身完整重建、同一U0语义恢复latent的解码。
均不是数学上界。预选视觉位置0、1、32、33；图文结果、失败帧和星座均保留，不挑好看的。

## 上传和安全解压安装

把压缩包与`.sha256`用已有SFTP上传到`/data1/zw/WYH/multimodel/`。
压缩包为源码/配置/文档/测试，不含权重数据。逐条执行，失败停在当前步骤：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_full_unfreeze_v1.tar.gz.sha256
test ! -e _v10_full_unfreeze_v1 && tar -xzf v10_full_unfreeze_v1.tar.gz
"$PYTHON_BIN" -B _v10_full_unfreeze_v1/install.py --project-root .
"$PYTHON_BIN" -B _v10_full_unfreeze_v1/install.py --project-root . --verify-only
```

安装器先核对所有前置源码哈希和冲突，再新增本模块；共享HANDOFF追加前备份原文。
相同内容重复安装可验证，不覆盖不同源码/旧结果；不删除失败目录，不执行全环境升级。
需要服务器保留完整V2 `experiment_results/v10_fso_fluctuating_v2_01/v10_screen`、B2大权重和同级`calibration`。
还依赖上一个`vae_tail_ablation_v1`已安装源码，不依赖其训练输出。

## 一条命令：检查→显存预检→配对训练→完整评价→打包

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
nvidia-smi -i 1 --query-gpu=name,memory.used,memory.free --format=csv
"$PYTHON_BIN" -B -u v10_extension/full_unfreeze_v1/launch.py pipeline \
  experiment_results/v10_full_unfreeze_v1_01 --max-hours 4
```

入口强制物理GPU1（程序内cuda:0），无需全局环境修改；两臂顺序运行，只驻留一个训练模型。
CPU检查必须通过，接着真实64对/每臂32更新预检覆盖骨干梯度、Adam状态和显存。
预检后重新从相同初始化开始主实验，预检更新不计入主实验。内存/梯度/权重/矩阵验收通过才能继续。
如果只想先看显存、不启动主实验：

```bash
"$PYTHON_BIN" -B -u v10_extension/full_unfreeze_v1/launch.py smoke \
  experiment_results/v10_full_unfreeze_smoke_01 --max-hours 0.5
```

仅CPU检查：`"$PYTHON_BIN" -B v10_extension/full_unfreeze_v1/launch.py check`。
默认启动要求空闲16GiB/磁盘12GiB，训练第1步及每20步采样保留4GiB显存余量。
这些是保护阈值，不保证永不OOM；采样之间其他任务也可能占显存。
OOM、非有限梯度、显存余量不足或超时：当前失败证据保留，停止后续任务，不自动改batch/精度/换卡/重试，不终止其他用户进程。
确需减显存的梯度检查点本版没有开启；避免同时引入额外兼容性变量。

如果V2不是`_01`，显式指定实际已完成目录，例如：

```bash
"$PYTHON_BIN" -B -u v10_extension/full_unfreeze_v1/launch.py pipeline \
  experiment_results/v10_full_unfreeze_v1_02 --max-hours 4 \
  --source-run experiment_results/v10_fso_fluctuating_v2_02/v10_screen
```

`_02`只是示例，要替换成真实存在的完整V2目录；可用`--calibration 实际目录`显式指定bank。
输出已存在时换新尝试编号，禁止删除/覆盖旧输出。没有自动断点续训。

## 耗时与进度

上一轮A6000尾部训练3000前向约12分钟，只能参考U1。本版每臂1000前向，U2新增编码/反传及审计开销尚未实测。
预检先留5–15分钟；完整流程暂留30–90分钟作规划范围，不是测量或保证。
GPU争用、I/O和全骨干反传会改变耗时；64步预检打印粗投影，主实验第100–200步日志更可靠。
每臂2小时/全流程4小时是安全停止上限，不是预计时长。

另开终端：

```bash
cd /data1/zw/WYH/multimodel
tail -n 30 experiment_results/v10_full_unfreeze_v1_01/v10_smoke.log
tail -n 30 experiment_results/v10_full_unfreeze_v1_01/v10_screen.log
```

记录每臂显卡型号、allocated/reserved峰值、采样空闲显存、参数资格/实际梯度覆盖、训练时间。
不要把PyTorch allocated误当nvidia-smi整进程占用。

## 完成后发什么

发回：`experiment_results/v10_full_unfreeze_v1_01/v10_full_unfreeze_v1_records.tar.gz`。
含摘要、完整图文/信道指标、实际图片和星座、训练/梯度/显存/诊断记录、可浏览`report.html`；大权重只保留服务器。
失败时发对应`v10_smoke.log`或`v10_screen.log`、子目录`FAILED.json`与已有`progress.json`，不重新启动碰运气。
本地没有Torch/真实骨干/COCO，语法或合成合同通过不等于真实训练、效果提升或论文结论。
