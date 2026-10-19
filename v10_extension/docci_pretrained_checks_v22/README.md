# V22：冻结模型的免训练辅助检查 + 图像数字基线

状态：2026-10-19代码交付。**不训练新融合模块；没有已测的V22模型收益。**
协议见 [PROTOCOL.md](PROTOCOL.md)。旧源码、权重、缓存和实验结果不覆盖。
先执行下面第1–2步，发回ASSETS.json；缺少新模型时不要安装/升级服务器依赖，也不要重复V20/V21。

## 1. 上传代码包，在现有项目解压

将代码tar.gz和同名.sha256一起上传 `/data1/zw/WYH/multimodel/`。包不含数据、权重或任何依赖安装程序。

```bash
cd /data1/zw/WYH/multimodel &&
sha256sum -c v10_docci_pretrained_checks_v22.tar.gz.sha256 &&
test ! -e v10_extension/docci_pretrained_checks_v22 &&
tar -xzf v10_docci_pretrained_checks_v22.tar.gz --keep-old-files &&
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_pretrained_checks_v22/test_core.py
```

CPU通过不代表真实模型通过；每个阶段自己独立验收，缺失、重复、错配、校验失败都不算完成。

## 2. 先查本地资产：不占GPU、不加载模型、不下载

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_pretrained_checks_v22/launch.py inventory \
  experiment_results/v22_inventory_01 --max-hours 0.25
```

发 `experiment_results/v22_inventory_01/ASSETS.json`。它列出现有版本及BART/VAE/LPIPS、BLIP-2、SD目录是否存在。
默认SD/BLIP-2路径只是候选，不假装模型已经在服务器；可以用 `--diffusion-path`/`--caption-path` 指定真实目录。
如文件不存在，下一步需在联网本机按具体版本下载权重再上传；本代码不会偷偷联网。
暂不提供猜测的权重版本/哈希或Linux BPG二进制。BPG未提供时，明确只跑JPEG参考，不把它写成BPG。

## 3. 已有V20资产可先做2图真实预检（无需新模型）

仅在GPU1是你有权用的空闲设备时运行，起始20GiB、运行4GiB空闲保护。不会停止别人任务。
下面只有冻结推理，真实旧权重锚点先重现；2图×clean/0/−5dB，每次单发，输出图片和实际接收文本。

```bash
cd /data1/zw/WYH/multimodel
CUDA_VISIBLE_DEVICES=1 /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_pretrained_checks_v22/launch.py prepare \
  experiment_results/v22_prepare_smoke_01 --profile smoke --max-hours 1
```

日志为 `experiment_results/v22_prepare_smoke_01.log`。日志出现Owned PID后，另开终端：

```bash
cd /data1/zw/WYH/multimodel
tail -f experiment_results/v22_prepare_smoke_01.log
```

准备耗时尚未测量；1小时是硬预算，不是ETA。前台命令可以等完；要后台运行，用已有nohup方式包住本命令，不要重复启动。
必须看到阶段独立verified、COMPLETE.json和LAUNCH.json exit_code=0；失败保留原目录，发日志，不改名重跑碰运气。

## 4. 图像数字JPEG参考（CPU，四MCS实际收发，不训练）

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_pretrained_checks_v22/launch.py digital \
  experiment_results/v22_jpeg_smoke_01 \
  --prepared experiment_results/v22_prepare_smoke_01 --codec jpeg --max-hours 1
```

JPEG低码率常需要降采样，记录实际尺寸；不据此断言连续胜过最强数字。
BPG需准备与服务器x86_64/Linux兼容的官方bpgenc/bpgdec，放项目内，并明确给SHA：
`--codec bpg --bpg-encoder 路径 --bpg-decoder 路径 --bpg-encoder-sha256 哈希 --bpg-decoder-sha256 哈希`。
缺二进制会停止，没有静默JPEG回退。

## 5. 新权重就绪后，封存一次再推理

`lock-assets`哈希本地所有文件，可能有磁盘读取时间；不复制大权重。模型来源/commit必须用你实际下载的值填写，不随便写main。
支持SD1/2标准目录；不支持SDXL/ControlNet。BLIP-2优先已有Transformers4.30兼容权重，具体选择需根据ASSETS报告确认。

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_pretrained_checks_v22/launch.py lock-assets \
  experiment_results/v22_sd_lock_01 --kind diffusion \
  --model-path pretrained/stable-diffusion-v1-5 \
  --origin-id YOUR_ACTUAL_MODEL_ID --revision YOUR_ACTUAL_COMMIT --max-hours 1
```

上面两项YOUR字段必须替换。来源不自动核验；训练集重叠未查明保持unknown。
BLIP-2同理，改为新输出、`--kind caption`、真实目录/来源/commit。

## 6. 两个互不自动触发的生成预检

先做文本→图像：原接收图像之外，输出同种子无提示/正确接收文本/错配接收文本三组。

```bash
cd /data1/zw/WYH/multimodel
CUDA_VISIBLE_DEVICES=1 /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_pretrained_checks_v22/launch.py enhance \
  experiment_results/v22_enhance_smoke_01 \
  --prepared experiment_results/v22_prepare_smoke_01 \
  --asset-lock experiment_results/v22_sd_lock_01/ASSET_LOCK.json --max-hours 2
```

图像→文本另跑caption；**不同时占GPU**。输出原接收文本、图像独立描述、正确/错配/空白图像+同一接收文本五组。

```bash
cd /data1/zw/WYH/multimodel
CUDA_VISIBLE_DEVICES=1 /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_pretrained_checks_v22/launch.py caption \
  experiment_results/v22_caption_smoke_01 \
  --prepared experiment_results/v22_prepare_smoke_01 \
  --asset-lock experiment_results/v22_blip_lock_01/ASSET_LOCK.json --max-hours 2
```

缺资产/显存/接口就停止，不能通过升级原环境或关闭严格检查凑过关。
数字图也能用同一enhance处理，另设输出并加：
`--receiver-stage experiment_results/v22_jpeg_smoke_01 --receiver-control jpeg_qpsk12`。
显式选一个MCS，不根据本批效果挑最好者；所有模态辅助开销也计入13376复符号。

## 7. 独立图像评分和直观展示

对prepare、digital、enhance各跑一次score，分别放新输出，只加载已准备的LPIPS资产：

```bash
cd /data1/zw/WYH/multimodel
CUDA_VISIBLE_DEVICES=1 /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_pretrained_checks_v22/launch.py score \
  experiment_results/v22_score_base_01 \
  --prepared experiment_results/v22_prepare_smoke_01 \
  --target experiment_results/v22_prepare_smoke_01 --max-hours 0.5
```

其他两次分别改target和新输出，如`v22_score_jpeg_01`/`v22_score_enhance_01`。未测LPIPS不填假数。
已有哪个阶段就列哪个；score必须同时列其target（prepare由--prepared提供）。例如只做JPEG时：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_pretrained_checks_v22/report.py render \
  --prepared experiment_results/v22_prepare_smoke_01 \
  --stages experiment_results/v22_jpeg_smoke_01 experiment_results/v22_score_base_01 \
  --output experiment_results/v22_display_smoke_01
```

展示固定前两图的全部条件，不选最好图；源图、接收图、生成图、文本和指标均标清条件。
`fact_review.csv`仅为人工审核表，不是自动算出的事实准确率；解盲键单独PRIVATE保存。

## 8. 发回轻量结果，不发权重

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_pretrained_checks_v22/collect.py \
  experiment_results/v22_smoke_01_records.tar.gz \
  --prepared experiment_results/v22_prepare_smoke_01 \
  --stages experiment_results/v22_jpeg_smoke_01 --max-hours 0.5
```

后续把实际完成的caption/enhance/score路径加入--stages；包自动带可离线打开的display/index.html。
包含PRIVATE真实图文，不公开GitHub；代码和聚合状态单独同步。大数组、模型目录和权重排除。
收集最多4GiB，失败不覆盖原数据，无旧版180秒硬超时；路径已存在需换collect02。

## 9. 只有真实smoke通过才扩大

prepare新输出用 `--profile quick --smoke-root experiment_results/v22_prepare_smoke_01`，8图；再pilot32图×3噪声实现。
caption/enhance也必须给对应实际成功的 `--smoke-root`，并使用同一asset-lock；不会从smoke自动扩大。
先回传2图预检，再决定扩大哪个方向。新耗时/峰值显存/效果尚未实测，不预支论文结论。
