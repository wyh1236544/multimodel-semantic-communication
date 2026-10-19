# V24：先校准 BLIP-2 基础看图，再决定是否继续互促

本版没有训练。复用服务器的BLIP-2及V23两图记录，先运行20行基础描述诊断。
不重复图像/文字传输，不使用原文纠错，不自动扩大或进入QA。详见[固定协议](PROTOCOL.md)。
原图仅作特权诊断；接收组只看接收图片。20行通常只需9次实际调用，其余为明确标注的同输入复用。
原V23、V20、V18代码、权重、数据、结果不覆盖。

## 1. 上传、校验、解压

将源码包和`.sha256`一起上传到现有`/data1/zw/WYH/multimodel`。本包不含权重、图片、数据集。
使用原服务器环境，不pip安装、不升级Torch/Transformers。

```bash
cd /data1/zw/WYH/multimodel &&
sha256sum -c v10_docci_caption_calibration_v24.tar.gz.sha256 &&
test ! -e v10_extension/docci_caption_calibration_v24 &&
tar -xzf v10_docci_caption_calibration_v24.tar.gz --keep-old-files &&
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_caption_calibration_v24/test_core.py
```

直接解压到`v10_extension/docci_caption_calibration_v24`，没有`install.py`。
如果目标已存在就停止，不能删除旧目录或覆盖重装。依赖固定当前V23及原153个历史文件，源码哈希不同会明确报错。

确认已有材料和空闲GPU：

```bash
cd /data1/zw/WYH/multimodel
ls -lh experiment_results/v23_inputs_smoke_01/COMPLETE.json experiment_results/v23_caption_lock_01/ASSET_LOCK.json
ls -lh BLIP2/blip2-opt-2.7b-2023/config.json
nvidia-smi -i 1 --query-gpu=index,name,memory.used,memory.free --format=csv
```

模型实际目录读取已验证的`ASSET_LOCK.json`，无需改名或搬动已有BLIP2目录。
起始空闲至少20GiB是保护下限，不是保证峰值；不要停他人进程。

## 2. 先只启动一次基础描述诊断

```bash
cd /data1/zw/WYH/multimodel
export V24_OUT=/data1/zw/WYH/multimodel/experiment_results/v24_caption_calibration_01
test ! -e "$V24_OUT" &&
test ! -e "${V24_OUT}_launch.log" &&
nohup /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_caption_calibration_v24/launch.py caption "$V24_OUT" \
  --prepared experiment_results/v23_inputs_smoke_01 \
  --asset-lock experiment_results/v23_caption_lock_01/ASSET_LOCK.json \
  --max-minutes 30 > "${V24_OUT}_launch.log" 2>&1 < /dev/null &
```

launcher内固定物理GPU1。先CPU测试、真实tiny随机HF CPU合同，再校验15.5GB已下载权重并推理。
校验期间每约10秒显示HASH进度；不会重复下载。64token上限不是完整长段落恢复目标。
30分钟为保护预算，不是运行时间预测；新配方耗时/峰值要以本次实际记录为准。

## 3. 看进度与完成状态

即使新终端丢了环境变量，也可使用下面的完整路径：

```bash
tail -n 60 /data1/zw/WYH/multimodel/experiment_results/v24_caption_calibration_01_launch.log
tail -n 40 /data1/zw/WYH/multimodel/experiment_results/v24_caption_calibration_01.log
```

`.contracts.log`是CPU测试，`.hf_contracts.log`是真实tinyHF CPU合同，`.log`是实际BLIP-2，
`.collect.log`是独立验收和私人打包。测试尚未完成时后面的日志可能尚不存在。
原生生成过程中每完成一行会打印序号、条件、新token数、空输出和复用标记。

```bash
/home/zw/anaconda3/envs/torch/bin/python -B -c \
"import json; from pathlib import Path; p=Path('/data1/zw/WYH/multimodel/experiment_results/v24_caption_calibration_01/LAUNCH.json'); print(p.read_text() if p.is_file() else '尚未写最终启动器状态，查看日志')"
```

只有`LAUNCH.json`为`VERIFIED`且records存在，才是完整工程完成。
`COMPLETE.json`仅说明推理结束；自然EOS不说明内容正确；空输出也会保留，不能当通过。

## 4. 发回哪个文件

正常完成后发：

`/data1/zw/WYH/multimodel/experiment_results/v24_caption_calibration_01_records.tar.gz`

内含PRIVATE网页、两图全部输出、原始token、EOS/概率/数值/显存峰值、原prepared副本与源码快照。
网页是`v24_records/index.html`，不是公开发布页面。不要上传权重或大缓存。
失败时先发`_launch.log`和`.log`/`.hf_contracts.log`对应报错，不删除失败目录、不重跑旧名字。

如果推理已完成、只有收集失败，可以CPU单独重收集，不再调用模型：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_caption_calibration_v24/collect.py \
  experiment_results/v24_caption_calibration_01 \
  experiment_results/v24_caption_calibration_01_records_collect02.tar.gz
```

## 5. QA是后续人工批准的单独阶段，不要现在运行

收集内容中`CAPTION_REVIEW_TEMPLATE.json`需要人工检查4个基础输出：两个原图和两个clean接收图。
填写reviewer、每项切题/主要物体有依据/无长循环，全部通过后decision才改为`APPROVE_QA_ONLY`。
这不是机器事实评分，也不是形式填表就能宣称有效；若任何输出不可靠，保持PENDING或拒绝。
保存为新的项目内文件，不改历史输出。通过后可运行：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_caption_calibration_v24/launch.py qa \
  experiment_results/v24_qa_calibration_01 \
  --prepared experiment_results/v23_inputs_smoke_01 \
  --asset-lock experiment_results/v23_caption_lock_01/ASSET_LOCK.json \
  --caption-root experiment_results/v24_caption_calibration_01 \
  --review experiment_results/v24_caption_review_01.json --max-minutes 30
```

QA共80行，通常36次实际短回答；没有长文本融合。没有同配方验证通过的人工审核就拒绝。
float32仅在有数值故障证据后独立新目录显式`--dtype float32`诊断，起始36GiB空闲；不自动执行。
没有训练/quick/pilot入口；后续扩大和互促训练需要看完本轮结果另行决定。
