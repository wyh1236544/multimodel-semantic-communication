# V10 原始 COCO 短描述 + 图片／文字单独传输 V2

本版本是新增模块，不覆盖 `bart_clip_joint_v1`、旧权重、结果、Localized
Narratives 标注或 `ln_v10_splits.json`。服务器离线，使用已有 torch 环境，
不安装依赖、不下载模型、不动其他人的 GPU 进程。

## 1. 这轮回答什么问题

原始 COCO 图像不变，将文字目标改回官方 COCO2017 短描述。仍使用原有固定
图片划分，只生成独立的、有哈希记录的 caption view。每图选择最小 annotation
ID 的非空描述；不是选择最短、最容易恢复或指标最高的描述。训练图片由固定
seed42 顺序取前256张，验证为既有 validation 前16张；没有访问最终 test。
种子仅固定数据顺序和随机过程，不做多种子扩大训练。

CLIP、BART-base、VAE、精确 B4 参考仍冻结；新增桥接层继承 V1 的同一组注册
参数。文字仍为 BART128位置 -> 有序T32压缩 -> 接收桥 -> 冻结BART自回归。
不换损失、贪心生成策略、EOS规则，不改内核来掩盖失败。BART特征对齐预热
之后继续CE+特征损失；图片使用latent+像素损失；公共分支使用原CLIP对齐。

默认分别独立训练与评价两种模式，不是把联合权重直接遮掉一个输入：

| 模式 | 编码时真正使用的输入 | 物理发送的槽 | 复符号数 | 恢复任务 |
|---|---|---|---:|---|
| image_only | 图片；VAE/B4图像特征、CLIP图像特征 | C32 + I224 | 2048 | 图片 |
| text_only | COCO短文本；BART、CLIP文本特征；不打开图片文件 | C32 + T32 | 512 | 文字 |
| joint（可选） | 图片和COCO短文本 | C32 + I224 + T32 | 2304 | 图片与文字 |

image_only 不调用BART/CLIP文本编码器；图像I使用 B4 separate 的图片独立路径，
其旧接口仅接收固定零文本占位，不接收实际描述。单模态的公共C只含该模态
信息，不能称为图文联合语义。不同模式使用的可训练路径不同，记录各模式
实际活跃参数；注册参数相同不代表相同活跃参数、计算量或功耗。

帧真正裁短：不发送未启用模态的零槽。接收端按预先公开固定模式还原布局。
模式属于双方已约定协议，不伪装为免费动态控制信令。接收端不获得原文、
原图、原文长度、源mask、CLIP教师向量；不存在UTF8旁路。生成文字没有被
再次输入图片解码器，这轮不是“用已恢复文字引导生成图片”的实验。

## 2. 先诊断，再独立评价

默认流水线先执行2条 TRAIN文本的极小smoke（8 forward），检查实际权重、
梯度、保存/重载、两种紧凑帧、GPU等能否执行，smoke不作为效果结论。
main每个模式使用各自新桥权重：

1. 图片：256 TRAIN x 4 epochs =1024 forward/512 optimizer updates，干净训练。
2. 文字：先对8条短文本做 T32-only 的干净诊断，1200 forward/600 optimizer
   updates，前200步特征预热。C固定零、只使用T32；仅用于与旧长文本瓶颈
   定位对照，不是256复符号的物理传输基线。
3. 文字实际C+T模式再从相同初始桥权重对8条做同预算诊断。正常、零接收
   记忆、错配接收记忆的实际生成用于判断有没有学会恢复。T32失败不自动
   否定C+T；C+T有更多资源，改善不能全归因于短文本。
4. C+T诊断合格后，从相同初始桥重置再做256 x4干净训练，不能将记住8条
   文本的诊断权重直接作为泛化权重。16 validation 做实际贪心恢复检查。
5. 通过该模式的工程检查后，在16 validation图像ID上评价 clean +6个FSO
   条件；每模式112行，默认两个模式都通过则224行。失锁帧计入，空文本/
   零图像不被剔除。可视化固定前4个ID，不按最好看的结果筛选。

文字诊断工程质量门槛 WER<=.20，置零/错配记忆WER比正常至少高.10；探索性
validation WER<=.60且同样有接收记忆依赖。未通过时保存实际失败文字并标记
`overfit_stop`/`clean_validation_stop`，不做该模式更大实验，图片模式独立保留。
数值错误、冻结权重变化、梯度/保存损坏、grad/no-grad前向不一致属于执行
失败，会停止并保存FAILED及日志，不被当作“换数据集没效果”。

**本轮只有干净训练 + FSO评价，不是新增带噪训练。** 这是先隔离短文本数据源
与单模态分支的功能；原V1带噪训练代码保留。不能声称该轮已证明噪声训练
收益、联合胜过分别传输、优于JPEG/数字链路，或满足最终论文泛化条件。

## 3. 光链路与资源

沿用已完成B4的真实全波形配置：64GSa/s、4SPS、16GBaud、RRC rolloff .1、
IQ-MZM、EDFA、FSO Gamma-Gamma/指向误差、接收噪声、ADC/DAC、同步导频、
跟踪/均衡。只发送一次，R=1；不是24/50次发送。弱/中/强湍流下的六个发射
参考功率点默认为 weak_p06、moderate_p18/p21/p24、strong_p15/p18。
预设的是光参数与参考功率，实际接收功率/符号SNR及导频SNR由波形计算。

以继承的默认硬件配置，含probe、前导、导频、滤波尾部和guard：

| 模式 | 实际调度AWG采样点 | 光波形调度时间 |
|---|---:|---:|
| image_only | 12026 | 0.18790625微秒 |
| text_only | 5114 | 0.07990625微秒 |
| joint | 13178 | 0.20590625微秒 |

这是波形采样调度时间，不包括传播L/c、NN处理、设备缓冲/驱动延迟，也不是
运行仿真的CPU时间。结果另存传播/资源字段，不能宣传为实机端到端时延。
三者共享峰值/总时隙上限，不保证相同实际能量；图文两次单独发送包含两套
probe等开销，不能直接当成与一次joint等资源的因果优越性对照。
连续特征无二进制BER/CRC，相关字段为null；不伪造bit/s、准确比特数或字节
压缩率。原文UTF8字节数仅记录业务源大小，不等于浮点特征线上的字节。

## 4. 上传、解压、安装

本地下载本次提供的压缩包及sha256文件至桌面。服务器IP/SSH别名换成你已有
的连接目标，不在此猜IP：

```bash
scp ~/Desktop/v10_bart_coco_modal_v2.tar.gz ~/Desktop/v10_bart_coco_modal_v2.tar.gz.sha256 zw@服务器IP:/data1/zw/WYH/multimodel/
```

服务器使用现有环境，逐段复制，不要复制终端提示符：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_bart_coco_modal_v2.tar.gz.sha256
test ! -e _v10_bart_coco_modal_v2 && tar -xzf v10_bart_coco_modal_v2.tar.gz
"$PYTHON_BIN" -B _v10_bart_coco_modal_v2/install.py --project-root /data1/zw/WYH/multimodel
```

安装器校验全部历史依赖和新增源码，再增量安装；旧文件哈希不匹配时拒绝
安装，不覆盖。更新共享HANDOFF前先做逐字节备份。完全相同的包可以重复
执行安装器核验，解压目录已存在时不再覆盖解压。

## 5. 运行前的数据检查（不占GPU）

必须已经有以下官方JSON；原Localized Narratives还在，不删除也不改名：

```text
COCO2017/annotations/captions_train2017.json
COCO2017/annotations/captions_val2017.json
```

```bash
"$PYTHON_BIN" -B v10_extension/bart_coco_modal_v2/launch.py data-check
```

如果缺JSON，把报错发来；不能用LN文件改名冒充，也不能在离线服务器自动
下载。`--data-root`可指向项目已有的正确COCO目录，固定split不能另建随机划分。

现有预训练依赖不新增：`pretrained/bart-base`（完整bin/tokenizer）、CLIP、VAE，
混合训练base权重、A delta、B2 delta、已完成非smoke B4结果和checkpoint。
默认路径继承V1；目录不同可用`--source-run`、`--legacy-run`、`--checkpoint`、
`--a-delta`、`--bart-path`等明确覆盖。不要绕过旧完成标记/权重哈希。

## 6. 默认一次跑图片与文字，物理GPU1

```bash
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used,memory.free --format=csv
"$PYTHON_BIN" -B v10_extension/bart_coco_modal_v2/launch.py pipeline experiment_results/v10_bart_coco_modal_v2_01 --max-hours 2
```

启动至少16GiB空闲，运行保留4GiB；不足就停止，不杀别人进程。FP32、batch1、
累积2，所有依赖维持现有环境。pipeline自动先CPU/Torch合同，再真实smoke，
再main，最后验收和打包。默认只跑两组，不加多种子。

若SSH窗口可能断开，改用以下启动方式，**不要前台和后台启动两次**：

```bash
test ! -e experiment_results/v10_bart_coco_modal_v2_01 && test ! -e experiment_results/v10_bart_coco_modal_v2_01_launch.log && nohup "$PYTHON_BIN" -B -u v10_extension/bart_coco_modal_v2/launch.py pipeline experiment_results/v10_bart_coco_modal_v2_01 --max-hours 2 > experiment_results/v10_bart_coco_modal_v2_01_launch.log 2>&1 &
```

可选仅一种：新目录加 `--modes image_only` 或 `--modes text_only`；可选联合
另开目录：

```bash
"$PYTHON_BIN" -B v10_extension/bart_coco_modal_v2/launch.py pipeline experiment_results/v10_bart_coco_modal_v2_joint_01 --modes joint --max-hours 2
```

第一轮建议默认两组。预计20–60分钟是规划范围，未经新版本实测；前一版
单条文字诊断全run约161秒，但不能等同于这轮图片+文字+完整FSO。2小时是
强制上限，不是结束时间保证。训练日志每20步给出该阶段预计剩余时间。

## 7. 查看进度与回传

```bash
tail -n 60 experiment_results/v10_bart_coco_modal_v2_01/v10_smoke.log
tail -n 80 experiment_results/v10_bart_coco_modal_v2_01/v10_screen.log
pgrep -af 'bart_coco_modal_v2/launch.py|bart_coco_modal_v2/run.py'
```

main开始后：

```bash
"$PYTHON_BIN" -m json.tool experiment_results/v10_bart_coco_modal_v2_01/v10_screen/progress.json
"$PYTHON_BIN" -m json.tool experiment_results/v10_bart_coco_modal_v2_01/v10_screen/mode_status.json
```

完成或质量停机后自动生成：

```text
experiment_results/v10_bart_coco_modal_v2_01/v10_bart_coco_modal_v2_records.tar.gz
```

发这个包即可。包含实际原文/恢复文本、原图/恢复图、六种链路、星座图、
PSNR/SSIM、WER/CER/Exact/BLEU2、截断/生成容量停止、实际光参数/资源、训练
和冻结审计。未传输模态显示“不适用”；图片描述只用于评价，不能作为恢复
文本展示。权重留服务器，轻量包只保存权重哈希。

`COMPLETE.json`=所有请求模式都完成；`PARTIAL_COMPLETE.json`=至少一种文字
模式未过质量门槛；`SMOKE_COMPLETE.json`=只有执行预检。验收器区分三种。
执行失败没有完成标记，此时发当前子日志、FAILED.json、learning_audit.json、
forward_consistency.json。修复后用02等新目录，保留01，不删失败证据。

手动重新打包（不会自动重跑训练）：

```bash
"$PYTHON_BIN" -B v10_extension/bart_coco_modal_v2/collect.py experiment_results/v10_bart_coco_modal_v2_01/v10_screen v10_bart_coco_modal_v2_records_manual.tar.gz
```

本地验证21项非Torch新合同+49项继承NumPy物理回归；真实Torch/HF的11项新增
数值合同、预训练权重、GPU显存、恢复效果仍由服务器smoke/main验证。本地无
Torch/模型，不能将源码语法和合成元数据测试当作模型有效的证明。
