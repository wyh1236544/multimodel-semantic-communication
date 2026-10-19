# V24 Fix2：严格保存半精度模型的 dtype 元数据

本次只修复工程兼容问题，无训练，不代表语义效果已经成功。

## 原因与边界

用户日志已证明完整模型加载后，原V24写MODEL.json时遇到实际`torch.dtype`。
这是新增代码漏测了半精度加载后的配置序列化，不是用户操作、显存或权重损坏。
仅凭traceback不能确定是哪一层配置中的哪个字段；新版本会在日志和FIX2.json列出实际路径。
Transformers的[to_dict实现](https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/configuration_utils.py)
也会把dtype转为文字；但固定4.30.2版本的
[Blip2Config.to_dict覆盖方法](https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/models/blip_2/configuration_blip_2.py)
直接复制顶层属性并返回，没有调用该转换函数。这解释了半精度加载后的顶层dtype可能遗留，
实际服务器字段路径仍以新日志为准。

新增模块依赖原V24及Fix1，184个旧源码固定不变。不覆盖任何旧源码/manifest/包或失败输出。
保留Fix1的内部语言模型EOS/PAD读取；生成函数仍是原模型函数。

只复制本次MODEL.json和FIX1.json元数据，将**真实torch.dtype**明确转换为如`float16`的文字，
记录每个转换路径。完整JSON预检发生在创建临时文件前。
不使用default=str；未知对象、tensor、NaN/Inf、循环结构、JSON键冲突仍报错。
不修改model.config、模型权重、tokenizer、生成参数、质量阈值、其他JSON写入规则。
FIX2.json/源码快照在原COMPLETE哈希封存前加入结果，原验收和Fix1验收继续执行。

## 1. 上传新包及SHA256文件后执行

旧目录保留，不能复用失败运行目录，不需要重新下载BLIP2或训练。
直接新增Fix2模块，没有install.py。

```bash
cd /data1/zw/WYH/multimodel &&
sha256sum -c v10_docci_caption_calibration_v24_fix2.tar.gz.sha256 &&
test ! -e v10_extension/docci_caption_calibration_v24_fix2 &&
tar -xzf v10_docci_caption_calibration_v24_fix2.tar.gz --keep-old-files &&
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_caption_calibration_v24_fix2/fix.py check
```

## 2. 启动新的两图caption诊断

```bash
cd /data1/zw/WYH/multimodel
export V24_FIX2_OUT=/data1/zw/WYH/multimodel/experiment_results/v24_caption_calibration_fix2_01
test ! -e "$V24_FIX2_OUT" &&
test ! -e "${V24_FIX2_OUT}_launch.log" &&
nohup /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_caption_calibration_v24_fix2/fix.py launch caption "$V24_FIX2_OUT" \
  --prepared experiment_results/v23_inputs_smoke_01 \
  --asset-lock experiment_results/v23_caption_lock_01/ASSET_LOCK.json \
  --max-minutes 30 > "${V24_FIX2_OUT}_launch.log" 2>&1 < /dev/null &
```

顺序：原28项CPU→Fix1的14项CPU→Fix2的8项CPU→**6项真实tinyHF**→完整BLIP2推理→验收/私人打包。
tinyHF包含原来的4项真实生成/配置测试，以及实际torch.dtype复制测试和
本地随机微型BLIP2保存再以float16加载的MODEL.json保存测试。
无网络/下载、无完整权重参与小模型测试、不跳过测试；CPU不做FP16前向，真实生成仍用FP32小模型。

仍物理GPU1，起始空闲20GiB，运行保留4GiB。仍20行/通常9次实际调用、64token greedy、无禁止重复。
30分钟是整个启动流程的硬预算，**不是实测预计耗时**；无自动重试/扩量/QA。
旧进程已经退出，完整BLIP2必须重新加载一次，但无需重跑历史训练和V23图片准备。

## 3. 看进度

```bash
tail -n 60 /data1/zw/WYH/multimodel/experiment_results/v24_caption_calibration_fix2_01_launch.log
tail -n 60 /data1/zw/WYH/multimodel/experiment_results/v24_caption_calibration_fix2_01.hf_contracts.log
tail -n 40 /data1/zw/WYH/multimodel/experiment_results/v24_caption_calibration_fix2_01.log
```

后两份日志按阶段创建；尚不存在就先看启动日志。
应先见`REAL_HF_FP16_METADATA_GATE_PASSED`，随后完整加载后见`FIX2 metadata preflight passed`，
再见`V24 caption 1/20 ...`。只有LAUNCH.json的status=VERIFIED且records存在才是工程完成。
facts仍PENDING、semantic_success仍false，必须审阅实际输出再讨论语义收益。

正常完成发回：

`experiment_results/v24_caption_calibration_fix2_01_records.tar.gz`

若再失败，先发新的`_launch.log`、`.hf_contracts.log`和`.log`；不要上传权重、不要自行升级环境。
只收集已完成的Fix2推理（不重跑模型）可用：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_caption_calibration_v24_fix2/fix.py collect \
  experiment_results/v24_caption_calibration_fix2_01 \
  experiment_results/v24_caption_calibration_fix2_01_records_collect02.tar.gz
```

本地无Torch/HF：8项新CPU回归使用明确TEST_ONLY替身验证生产流程，并非实际模型推理通过。
实际服务器的6项tinyHF与完整权重运行仍是强制门禁。GitHub只保存源码/说明，不上传私人图文/records/权重。
