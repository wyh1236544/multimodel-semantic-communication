# V23：免训练证据检查

只新增本目录。保留V22和所有旧源码/缓存/模型/结果。请先运行下面的CPU两图对照，
不要重新训练V14/V18/V20，不安装或升级服务器环境。默认没有任何自动下一阶段。
本机CPU合同与真实服务器预训练/GPU结果分开；交付状态见公共docs/results中的V23说明。

## 1 上传、解压和CPU测试

把代码包和.sha256上传到现有multimodel目录。V22及其原有依赖必须已经安装。
新包直接解压到v10_extension/docci_evidence_v23，没有install.py。

```bash
cd /data1/zw/WYH/multimodel &&
sha256sum -c v10_docci_evidence_v23.tar.gz.sha256 &&
test ! -e v10_extension/docci_evidence_v23 &&
tar -xzf v10_docci_evidence_v23.tar.gz --keep-old-files &&
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_evidence_v23/test_core.py
```

测试中的合成图/文本不是实验成果。测试通过后先做无模型资产检查：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_evidence_v23/launch.py inventory \
  experiment_results/v23_inventory_01 --max-hours 0.5
```

## 2 现在就能跑：复用V22两图，不重复GPU生成

下面先CPU导入已完成的真实V22 smoke，再CPU完成JPEG四MCS实际收发。
所有PNG/文字沿用旧文件、旧原目录只读，噪声也沿用V22，不会与连续结果错配。

```bash
cd /data1/zw/WYH/multimodel &&
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_evidence_v23/launch.py import-v22 \
  experiment_results/v23_inputs_smoke_01 \
  --from-v22 experiment_results/v22_prepare_smoke_01 --max-hours 0.5 &&
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_evidence_v23/launch.py digital \
  experiment_results/v23_jpeg_smoke_01 \
  --prepared experiment_results/v23_inputs_smoke_01 --max-hours 1
```

日志为输出目录名加.log；独立CPU重放为.verify.log，重放阶段也会显示进度。
1小时是停止上限，不是实测耗时。不要对同一目录重复启动。
数字阶段CPU运行，并独立重算PSNR。原始连续图的LPIPS已在V22算过，以下先CPU导入同一评分，不重复占GPU。
只有JPEG图的LPIPS需要GPU1空闲，不与其他GPU任务并发。

```bash
cd /data1/zw/WYH/multimodel &&
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_evidence_v23/launch.py import-v22-score \
  experiment_results/v23_score_base_smoke_01 \
  --from-v22 experiment_results/v22_prepare_smoke_01 \
  --from-v22-score experiment_results/v22_score_base_01 \
  --prepared experiment_results/v23_inputs_smoke_01 \
  --target experiment_results/v23_inputs_smoke_01 --max-hours 0.5 &&
CUDA_VISIBLE_DEVICES=1 /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_evidence_v23/launch.py score \
  experiment_results/v23_score_jpeg_smoke_01 \
  --prepared experiment_results/v23_inputs_smoke_01 \
  --target experiment_results/v23_jpeg_smoke_01 --max-hours 0.5
```

## 3 发回两图数字检查结果

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_evidence_v23/collect.py \
  experiment_results/v23_image_smoke_01_records.tar.gz \
  --prepared experiment_results/v23_inputs_smoke_01 \
  --stages experiment_results/v23_jpeg_smoke_01 \
           experiment_results/v23_score_base_smoke_01 \
           experiment_results/v23_score_jpeg_smoke_01 --max-hours 1
```

把v23_image_smoke_01_records.tar.gz下载给我。里面有PRIVATE图片、指标、网页和验收，
不含大权重。若先只完成JPEG，可省掉两个score路径，但LPIPS会明确未测。
如果失败，发本轮.log及.verify.log，不删失败目录、不直接扩到pilot。

## 4 BLIP-2单独准备：现在服务器缺权重，不能直接执行caption

下载脚本仅用标准库、分块读取，默认只显示计划、不访问网络。
在能联网的本地电脑先运行计划，路径替换为本地代码位置：

```bash
python3 v10_extension/docci_evidence_v23/download_caption.py \
  --output ./blip2-opt-2.7b-2023
```

完整官方权重约15.5GB（约14.4GiB），须有相应剩余磁盘；不是需要15.5GB常驻RAM。
本版不擅自改为其他较小模型。决定下载后，在同一命令加--download：

```bash
python3 v10_extension/docci_evidence_v23/download_caption.py \
  --output ./blip2-opt-2.7b-2023 --download --max-hours 12
```

打印逐文件百分比、大小及VERIFIED，最终出现All model files VERIFIED。
中断保留.part，手动执行同命令仅续同一固定commit，已完成文件校验复用。
没有服务器自动下载、绕过证书或无限重试。随后用你现有上传方式把完整目录放到
服务器pretrained/blip2-opt-2.7b-2023，包含DOWNLOAD_RECEIPT.json；不必再压缩复制15GB。
不能只上传两份bin漏掉tokenizer/config，不能从当前main混搭配置。

```bash
cd /data1/zw/WYH/multimodel &&
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_evidence_v23/launch.py lock-caption \
  experiment_results/v23_caption_lock_01 \
  --caption-path pretrained/blip2-opt-2.7b-2023 --max-hours 1 &&
CUDA_VISIBLE_DEVICES=1 /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_evidence_v23/launch.py caption \
  experiment_results/v23_caption_smoke_01 \
  --prepared experiment_results/v23_inputs_smoke_01 \
  --asset-lock experiment_results/v23_caption_lock_01/ASSET_LOCK.json --max-hours 2
```

这里只生成六组推理，不训练；2小时是硬预算不是ETA，真实预检后再估算。
BLIP-2内部视觉尺寸可能是224，不能称512高清细节全部保留。
收集时在第3节--stages中增加experiment_results/v23_caption_smoke_01，换新collect02包名。
若caption-only更好，只能说图像能提供一些信息；联合必须超过两个单路和错配/空图对照，事实也要审核。

## 5 只有实际smoke通过后，才手动扩展曲线

先quick8图×1draw，新网格clean/−8/−5/−2/0/2/5/10/20。
只有clean/0/−5另生成文本；其他点只走图像，避免无关文本生成耗时。
下面给出准备与数字命令，**不自动执行**：

```bash
cd /data1/zw/WYH/multimodel &&
CUDA_VISIBLE_DEVICES=1 /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_evidence_v23/launch.py prepare \
  experiment_results/v23_inputs_quick_01 --profile quick \
  --smoke-root experiment_results/v23_inputs_smoke_01 --max-hours 2 &&
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_evidence_v23/launch.py digital \
  experiment_results/v23_jpeg_quick_01 \
  --prepared experiment_results/v23_inputs_quick_01 \
  --smoke-root experiment_results/v23_jpeg_smoke_01 \
  --smoke-prepared experiment_results/v23_inputs_smoke_01 --max-hours 3
```

score/collect同前，替换成quick新路径。JPEG是低码率参考，不因结果好就宣称优于所有数字方案。
pilot32×3draw仍是固定dev；须另行手动profile pilot新目录，先看quick耗时/图像再决定，不自动升阶。
扩大caption必须附同资产的--smoke-root及--smoke-prepared，stage仍单独运行。
没有把成功caption当作下一步训练授权。

## 6 可选潜变量校准：不跑扩散、不训练

```bash
cd /data1/zw/WYH/multimodel
CUDA_VISIBLE_DEVICES=1 /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_evidence_v23/launch.py calibrate \
  experiment_results/v23_latent_calibration_smoke_01 --max-hours 1
```

固定8条TRAIN×3条件，量化误差结构与同VAE往返损失，不读取dev来选择扩散步数。
没有UNet、图文互促或扩散效果结论，NPZ留服务器；单独收集：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_evidence_v23/collect.py \
  experiment_results/v23_latent_calibration_smoke_01_records.tar.gz \
  --stages experiment_results/v23_latent_calibration_smoke_01 --max-hours 1
```

## 查看进度与完成条件

```bash
tail -n 50 /data1/zw/WYH/multimodel/experiment_results/v23_jpeg_smoke_01.log
tail -n 50 /data1/zw/WYH/multimodel/experiment_results/v23_jpeg_smoke_01.verify.log
```

看到V23 VERIFIED与LAUNCH.json status=VERIFIED才代表该阶段执行/验收完成。
只看到COMPLETE.json不等于独立验收通过，更不等于恢复质量成功。
失败时也保留所有已有记录和日志，不删除后重试到通过。
