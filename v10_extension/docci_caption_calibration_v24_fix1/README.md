# V24 Fix1：读取真正执行生成的语言模型配置

状态：修复接口兼容性，不是训练或恢复效果改进。旧V24所有源码/manifest/包、失败日志和历史结果不修改。

## 已确认问题与修复范围

用户服务器的两个真实tinyHF测试都在读取`model.generation_config.eos_token_id`失败。
Transformers4.30.2的BLIP-2外层`generation_config`可以为None；真正的生成在内部OPT语言模型执行。
原`runtime.py`的EOS/PAD读取和原`run.py`的MODEL.json配置序列化都有同一假设错误。
本次一起修复，不能只改EOS然后等主程序再失败。这是新增代码的兼容遗漏，不是显存或训练失败。
按原launcher顺序，失败发生在完整权重加载前；没有新的真实两图推理需要恢复。

固定版本依据：

- [BLIP-2 v4.30.2 generate](https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/models/blip_2/modeling_blip_2.py)：调用内部`language_model.generate`。
- [PreTrainedModel v4.30.2](https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/modeling_utils.py)：`generation_config`按`can_generate()`初始化，外层可能为空。

新增小型进程内只读ModelView：配置读取始终指向`language_model.generation_config`，
原`generate`仍绑定原BLIP-2对象。没有修改模型对象属性/配置、tokenizer、权重、EOS、生成kwargs或Torch全局。
缺失真实内部配置就报错，不编造默认EOS/PAD、不跳过检查。作用域退出恢复原V24函数绑定。
旧V24源码不落盘改写；新运行包含FIX1.json和独立repair源码快照，collector验证修复身份。
不是放宽长度/重复/EOS/质量标准，无训练、无环境升级、无重新下载。

## 1. 上传、校验、解压并检查

仅上传新修复包和SHA256文件到现有multimodel。不要重新解压覆盖原V24，也不要手动改runtime.py。

```bash
cd /data1/zw/WYH/multimodel &&
sha256sum -c v10_docci_caption_calibration_v24_fix1.tar.gz.sha256 &&
test ! -e v10_extension/docci_caption_calibration_v24_fix1 &&
tar -xzf v10_docci_caption_calibration_v24_fix1.tar.gz --keep-old-files &&
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_caption_calibration_v24_fix1/fix.py check
```

需要原V24完整安装，全部178个固定依赖必须匹配；若之前手动编辑过原文件，会停止。
解压直接新增`v10_extension/docci_caption_calibration_v24_fix1`，没有install.py。
CPU检查通过不是实际HF/GPU通过；launcher仍必须执行真实HF小模型测试。

## 2. 使用新的运行名

```bash
cd /data1/zw/WYH/multimodel
export V24_FIX_OUT=/data1/zw/WYH/multimodel/experiment_results/v24_caption_calibration_fix1_01
test ! -e "$V24_FIX_OUT" &&
test ! -e "${V24_FIX_OUT}_launch.log" &&
nohup /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_caption_calibration_v24_fix1/fix.py launch caption "$V24_FIX_OUT" \
  --prepared experiment_results/v23_inputs_smoke_01 \
  --asset-lock experiment_results/v23_caption_lock_01/ASSET_LOCK.json \
  --max-minutes 30 > "${V24_FIX_OUT}_launch.log" 2>&1 < /dev/null &
```

顺序：原28CPU检查→修复CPU检查→4项真实tinyHF检查→完整BLIP-2两图→独立CPU验收/PRIVATE打包。
tinyHF包含原两个失败方法，不删除、不SKIP。新增检查确认外层仍为None、内部EOS/PAD/配置读取正确、权重/钩子不变。
原来`v24_caption_calibration_01`即使只有日志、没有目录，也保留，不重用旧名字。

仍物理GPU1、起始20GiB空闲/运行4GiB余量、无训练、原20行/通常9次调用、64token greedy、无重复禁令。
30分钟为硬预算，不是ETA；不自动QA/扩量、不停止其他任务。

## 3. 查看日志与发回结果

```bash
tail -n 60 /data1/zw/WYH/multimodel/experiment_results/v24_caption_calibration_fix1_01_launch.log
tail -n 60 /data1/zw/WYH/multimodel/experiment_results/v24_caption_calibration_fix1_01.hf_contracts.log
tail -n 40 /data1/zw/WYH/multimodel/experiment_results/v24_caption_calibration_fix1_01.log
```

按顺序创建日志，后面的日志还不存在时先看启动日志。
只有LAUNCH.json为VERIFIED且records存在才是完整工程完成；语义成功始终false、factsPENDING，结果需人工审阅。

正常完成发：

`/data1/zw/WYH/multimodel/experiment_results/v24_caption_calibration_fix1_01_records.tar.gz`

若再次在HF检查失败，发新的`.hf_contracts.log`和`_launch.log`，不用上传权重，不自行改依赖。
原V23输入、BLIP2目录及封存资产全部复用；仍禁止把PRIVATE图文/records/权重发布GitHub。

## 4. 后续收集/QA仍走修复入口

只收集已完成推理，不重跑模型：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_caption_calibration_v24_fix1/fix.py collect \
  experiment_results/v24_caption_calibration_fix1_01 \
  experiment_results/v24_caption_calibration_fix1_01_records_collect02.tar.gz
```

本轮先只caption。原V24的人工QA审核规则和协议不改；审核通过后的QA将`launch caption`换成
`launch qa`并传相应caption-root/review，使用新输出路径，不绕过人工审核。

本地无Torch/HF：CPU对象/生产配置读表达式回归不等于真实HF推理通过；真实服务器测试仍是门禁。
