# V25：BLIP-2视觉特征直接辅助文本

这是新架构的小规模验证，**尚未得到真实V25效果**。不再要求BLIP先生成描述再纠错。
冻结BLIP2输出32×768视觉token，约152万参数的旁路适配器接到冻结BART末两层。
原图文codec/主干不动，不增加信道符号，但视觉侧多一次BLIP2计算；同规模适配器不代表全系统算力相同。
不重复训练V14/V18/V20，不安装依赖、重新下载权重或停止别人的任务。
本轮只实现图像→文字；文本→图片恢复与双向循环留作独立阶段。[固定实验协议](PROTOCOL.md)。

## 1 上传、解压、CPU检查

把新包及.sha256传到已有multimodel目录，保留旧版本及结果。需要V24 Fix2和其原有依赖已安装。
新包直接新增本目录，没有install.py。

```bash
cd /data1/zw/WYH/multimodel &&
sha256sum -c v10_docci_blip_feature_rx_v25.tar.gz.sha256 &&
test ! -e v10_extension/docci_blip_feature_rx_v25 &&
tar -xzf v10_docci_blip_feature_rx_v25.tar.gz --keep-old-files &&
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_blip_feature_rx_v25/test_core.py
```

合成CPU测试通过不是实际模型通过。可先运行不加载权重、不占GPU的路径清单：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_blip_feature_rx_v25/launch.py inventory \
  experiment_results/v25_inventory_01 --max-hours 0.5
```

默认复用：V20 `experiment_results/v20_text_guided_image_fix1_01`，V18 `v18_text_noise_01/train_noise_replay/last.pth`，
V24 `v24_caption_calibration_fix2_01`，BLIP2封存 `v23_caption_lock_01/ASSET_LOCK.json`。
真实权重路径从封存读取，即你已有的`BLIP2/blip2-opt-2.7b-2023`，不使用过时的pretrained/blip目录。
BART/VAE/LPIPS沿用已有路径。路径存在不等于字节正确，后面仍核验。

## 2 现在先只运行smoke

```bash
cd /data1/zw/WYH/multimodel
export V25_OUT=/data1/zw/WYH/multimodel/experiment_results/v25_blip_feature_smoke_01
test ! -e "$V25_OUT" &&
test ! -e "${V25_OUT}_launch.log" &&
nohup /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_blip_feature_rx_v25/launch.py smoke "$V25_OUT" \
  --max-hours 2 > "${V25_OUT}_launch.log" 2>&1 < /dev/null &
```

顺序：纯CPU合同→原V24的6项真实HF→V25的8项真实模型合同→8TRAIN/2dev缓存→self/visual各3更新→48行实际评价→独立CPU验收/PRIVATE收集。
8项新真实合同含原两项BLIP生成、Q-Former与native generate一致性、不调用OPT的特征提取、真实BART旁路等价、所有新组梯度/更新、磁盘保存、批间隔离和异常清理。
smoke不用于判断是否有论文增益，也不续用这6个更新的权重。2小时只是全流程硬上限，不是预计耗时。

```bash
tail -n 60 /data1/zw/WYH/multimodel/experiment_results/v25_blip_feature_smoke_01_launch.log
tail -n 50 /data1/zw/WYH/multimodel/experiment_results/v25_blip_feature_smoke_01/cache.log
tail -n 30 /data1/zw/WYH/multimodel/experiment_results/v25_blip_feature_smoke_01/train_visual.log
tail -n 30 /data1/zw/WYH/multimodel/experiment_results/v25_blip_feature_smoke_01/evaluation.log
```

后面日志按阶段创建，还没出现就先看启动日志。看到LAUNCH_FINISHED.json的status=VERIFIED且records存在，才是完整流程完成：

```bash
cd /data1/zw/WYH/multimodel
test -f experiment_results/v25_blip_feature_smoke_01/LAUNCH_FINISHED.json &&
test -f experiment_results/v25_blip_feature_smoke_01_records.tar.gz &&
/home/zw/anaconda3/envs/torch/bin/python -B -c \
"import json; print(json.load(open('experiment_results/v25_blip_feature_smoke_01/LAUNCH_FINISHED.json')))"
```

完成发`experiment_results/v25_blip_feature_smoke_01_records.tar.gz`。大缓存、适配器权重留服务器。
失败先发`_launch.log`及对应阶段日志末160行，不删失败目录、不要反复启动或自行升级环境。

## 3 实际smoke审核后，再手动pilot（不自动执行）

```bash
cd /data1/zw/WYH/multimodel
export V25_PILOT=/data1/zw/WYH/multimodel/experiment_results/v25_blip_feature_pilot_01
test ! -e "$V25_PILOT" &&
test ! -e "${V25_PILOT}_launch.log" &&
nohup /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_blip_feature_rx_v25/launch.py pilot "$V25_PILOT" \
  --smoke-root experiment_results/v25_blip_feature_smoke_01 \
  --max-hours 8 > "${V25_PILOT}_launch.log" 2>&1 < /dev/null &
```

512TRAIN/32重复使用的dev，每组512更新、4轮，两个新适配器总1024更新。固定末次权重，不挑验证集最优。
只训练小适配器，使用缓存的clean/0dB接收观测；噪声跨epoch复用，不是重新端到端训练图文codec。
新pilot重新初始化并独立缓存所需接收观测，不把smoke训练接进主实验。5种图文SNR组合、1280行输出，包括错图、灰图、清零、旁路关闭。
8小时是保护预算。训练ETA仅剩余当前组训练，不含后续自由生成/校验。新缓存和BART多次生成可能比训练本身耗时，真实smoke回传后再估总时间。
不得仅按两图BLIP推理32秒外推全流程。

显存未实测：保持GPU1启动至少20GiB空闲/运行4GiB余量；不是保证需要20GiB。
缓存阶段BLIP2和已有冻结主干同时在GPU，不能把V24单独BLIP的7.04GiB当本版峰值。
训练/评价阶段只加载BART及小适配器，不让大BLIP常驻。项目启动要求16GiB磁盘，主缓存预计数GiB（估算，压缩占用未测）。

主结果必须正确视觉辅助优于同规模self及错图；最终还要事实审核优于文本单路和caption-only，WER不能代替语义。
raw图文和网页PRIVATE；GitHub仅源码/协议/聚合。`semantic_success=false`和`facts=PENDING`在自动报告中保留。

## 4 单独重新收集已完成结果（不训练）

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_blip_feature_rx_v25/collect.py \
  experiment_results/v25_blip_feature_smoke_01 \
  experiment_results/v25_blip_feature_smoke_01_records_collect02.tar.gz
```

本地只具备NumPy/PIL，无Torch/HF。交付的纯CPU TEST_ONLY流程不代表新模型推理/训练已通过。
真实tinyHF、已保存父权重锚点、实际梯度/保存及完整GPU smoke是服务器强制门禁，不能SKIP。
