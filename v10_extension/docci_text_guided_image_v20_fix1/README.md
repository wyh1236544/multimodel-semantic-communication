# V20 Fix1：严格确定性池化兼容修复

状态：源码修复与本地 CPU 验证；本机无 Torch/HF/GPU，真实 CUDA 门禁和保存权重 smoke 须服务器执行。
本修复不改变 V20 研究协议、三组架构/参数、训练数据、损失、资源、信道或质量阈值。

## 原因与最小改动

首次服务器运行只完成 8+2+2 小缓存；A_self 首次反向传播失败，没有完成任何 optimizer update。
Fusion 的 adaptive_avg_pool2d CUDA backward 与 PyTorch 1.12 严格确定性设置不兼容。
官方依据：[PyTorch v1.12.0 对应实现](https://github.com/pytorch/pytorch/blob/v1.12.0/aten/src/ATen/native/cuda/AdaptiveAveragePooling.cu#L722-L728)。

只对原 model 模块的局部 F 别名提供固定分块均值，不改全局 Torch 函数，不改原 Fusion 的其余代码。
16/32/64 空间尺寸均整除 4×8 网格，数学定义和导数相同；浮点求和顺序可能产生末位差异。
保留 requires_grad 和严格确定性，既不 detach 自身特征，也不设置 warn_only/关闭确定性。
原 V20 源码、manifest 和失败目录全部保持原样。所有新产物明确带 Fix1 身份，不称精确断点续训。

## 上传、解压（无需再次上传 LPIPS 权重）

只上传 v10_docci_text_guided_image_v20_fix1.tar.gz 及其 .sha256 到现有项目根目录。
原 V20 和 _v20_lpips_assets 必须仍在，不升级或安装任何服务器依赖。

~~~bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_docci_text_guided_image_v20_fix1.tar.gz.sha256 &&
test ! -e v10_extension/docci_text_guided_image_v20_fix1 &&
tar -xzf v10_docci_text_guided_image_v20_fix1.tar.gz --keep-old-files
~~~

任何一步失败就停止，不删除旧目录重试、不手改原 manifest。

## 只启动一次：CUDA 门禁 → 复用小缓存 → 三组 smoke → 原定完整训练

只运行一个 pipeline；不要同时启动旧 V20 或另一个 smoke。GPU1 起始至少 20GiB 可用，运行保留4GiB，磁盘至少64GiB。
48小时沿用原预算，包含收集预留；不是预测。旧粗估20–40小时尚无真实训练速率支撑，以新日志为准。

~~~bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
export V20_OUT=experiment_results/v20_text_guided_image_fix1_01
test ! -e "$V20_OUT" &&
test ! -e "${V20_OUT}_launch.log" &&
test ! -e "${V20_OUT}_fix1_gpu_gate.json" &&
nohup "$PYTHON_BIN" -B -u \
  v10_extension/docci_text_guided_image_v20_fix1/launch.py pipeline "$V20_OUT" \
  --reuse-smoke-cache experiment_results/v20_text_guided_image_01/smoke_cache \
  --max-hours 48 > "${V20_OUT}_launch.log" 2>&1 < /dev/null &
~~~

缓存复用先按原规则校验源图/数组身份，独立复制约12个样本的文件到新输出，再逐字节/数组校验；
不重新提取这12个样本的特征，不动原失败目录，不硬链接或符号链接。
原来正式8192对缓存没有执行过，仍须按计划首次生成，不能复用12例冒充正式缓存。

门禁依次检查：原CPU/Torch合同、新CPU合同、3项真正CUDA合同
（各尺度池化前向/梯度等价和重复一致、三组端到端反向更新、LPIPS梯度与冻结）。
CUDA门禁不得跳过，成功打印 FIX1_STRICT_CUDA_GATE_PASSED，保存独立回执；
之后真实父权重/VAE/BART/Long-CLIP/LPIPS smoke 仍全量执行。任一步失败即停止，不自动重试。
入口没有远程控制功能，由用户在现有服务器环境运行，不会终止他人作业。

如果只想先跑短预检，把 pipeline 改为 smoke、使用不同全新输出，并把 --max-hours 改为2；
短预检不会自动开始正式训练。不要和上面的完整 pipeline 同时运行。

## 查看进度

~~~bash
tail -n 60 experiment_results/v20_text_guided_image_fix1_01_launch.log
tail -f experiment_results/v20_text_guided_image_fix1_01/train_B_tx_text.log
~~~

完整训练日志尚未生成时，看总日志及 smoke_train_A_self.log 等；之后主顺序仍 B/C/A。
CUDA门禁发生在新结果目录创建前，因此若门禁失败，总日志可能是唯一新产物；保留并发回。

## 结束后发什么

自动生成的 PRIVATE 轻量包及 SHA：

~~~text
experiment_results/v20_text_guided_image_fix1_01_records.tar.gz
experiment_results/v20_text_guided_image_fix1_01_records.tar.gz.sha256
~~~

大数组/权重继续留服务器。手动验收或收集必须经过 Fix1 入口，不能使用原 collector：

~~~bash
"$PYTHON_BIN" -B -u v10_extension/docci_text_guided_image_v20_fix1/entry.py collect \
  experiment_results/v20_text_guided_image_fix1_01 \
  experiment_results/v20_text_guided_image_fix1_01_records_collect02.tar.gz \
  --max-minutes 60
~~~

新增运行源码快照、CUDA门禁回执和每阶段FIX1_RUNTIME纳入记录；原指标计算/置信区间/质量门槛不变。
真实 DOCCI 图文、records、网页和权重仍私有，不上传公共 GitHub。
