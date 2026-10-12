# Text recovery V3：先修文字恢复，再回到图文联合

状态：代码与本地非Torch检查完成后发布；真实Torch/预训练GPU效果由服务器
检查。不是已经解决文字问题，也不承诺收益。旧V1/V2、权重、数据、固定图片
划分及全部结果保留。本版本是独立文本修复实验，不重训图片，不删除C/I/T。

## 本轮的五个对照

|分支|改动|实际载荷复符号|
|---|---|---:|
|A0_legacy|原式T32x16、128位置全可见、旧损失/LR|256|
|A1_length_mask|仅改为实际接收长度的注意力掩码；其他训练策略同A0|264|
|A2_ramp|A1＋200步预热后200步渐增CE；特征权重1、pad权重.1、桥LR1e-4|264|
|A3_lora|A2＋BART解码器最后两层cross-attention Q/V rank8 LoRA；LR5e-5|264|
|A4_wide|A3把每个T槽16维改为64维，定位容量问题|1032|

A0是从新初始化训练的原式私有T32接口对照，不是重载V2旧权重的精确历史
成绩复现。保留原私有诊断中的零C接收偏置/注意力路径，不提供任何未发送的
Common源信息。窄分支的初始桥张量一致；A4投影形状/容量/资源不同。

A3增加49,152可训练LoRA参数，并非全解冻139M BART；原BART参数仍使用且
占显存。CLIP全冻结；VAE/B4不加载，不打开图片。A2-A1是训练策略组合，不能
把收益归到其中一个单独因素；A3-A2为同载荷/数据/步数的轻量解码器适配
比较，但额外算力与参数不假装相同。A4不能用来宣称同带宽收益。

可选`--include-common`添加A5_common：C32+T32+帧头=520复符号。此处C仅有
文字信息，还不是图文融合；不是264复符号A3的严格等资源公共分支收益。
本轮诊断成功后再将修好的文本路径接回C/I/T，做真正同资源联合消融。

## 长度不是免费信息

源句包含BOS/EOS，长度2..128。发送(length-1)的7个比特，每比特用两个±1实
符号，再加2个固定公开padding实符号：共16实/8复符号，放入同一光帧。
不传原文UTF8，不免费传源mask。接收端仅从收到的帧头合并判决长度，生成
自己的mask；无CRC，错误长度可能被接受，结果明确报告长度错误与空接收。
这不是整帧重复发送；语义特征仍只发送一次。头被损伤时不会用真长度修正。

Native原生masked/unmasked/pad-zero对照可使用源mask，但明确仅为TRAIN
oracle诊断，不是部署方案或等资源基线。先核对原编码器与缓存特征接口、
整句teacher与增量缓存的逐步logits；不一致时执行失败停止。Native若质量
未过.20，增加一个600步仅LoRA的8句诊断；其拟合参数绝不用于生产初始化。

## 训练与停止条件

默认只做五组TRAIN8短句诊断，各1200forward/600optimizer，FP32/batch1/
累积2，一个固定seed42、不扫种子。前200forward特征预热，A2以后CE逐渐
增加。每200步保存前2条实际贪心输出；不是用teacher-forcing准确率代替
接收生成。最终完整8条正常/零记忆/错配记忆都保存，不禁止重复或强制EOS。

WER<=.20、零/错配比正常至少差.10的旧门槛保留，并新增必须自然结束。
8/8正确是期望目标，不把这个小样本记忆当泛化能力。任何失败继续保留原文
与错误输出；执行错误/冻结变化/梯度或重载失败则立即停止，不重试。

`diagnostic`先CPU/HF小模型测试，再实际2句8步smoke，再正式五组诊断。
`SCREEN_COMPLETE`只表示诊断流程跑完，**不表示恢复通过**；Native失败会
标native_stop，各分支质量失败会标overfit_stop。`SMOKE_COMPLETE`只表示
可执行，不代表学习质量。没有把质量失败包装成主实验COMPLETE。

可选`pipeline`执行同样诊断，仅预先约定的A2/A3在各自过门槛后，从未拟合
8句的原始初始化重新训练256文本x8epochs=2048forward/1024optimizer。
验证固定32个原validation IDs，WER<=.60、同样记忆依赖和自然结束后才做
六种完整FSO全波形评价。验证是探索性，可能历史已用，不是最终test。
本轮只有干净训练，**没有新增FSO带噪训练**。质量未过时不会扩大到物理评价。

光链路读取已完成B4实验design中的原配置，沿用64GSa/s、4SPS、16GBaud、
RRC.1、V3导频同步、光功率/噪声/Gamma-Gamma/指向模拟。真正发送新的紧凑
帧及帧头，不裁切旧2304代理bank。probe/preamble/pilot/tails/guards全计费；
资源时间是AWG调度时间，不是NN/传播/实机端到端时间。保留所有失锁帧、
实际光功率/导频SNR/符号SNR/能量/文字/星座图。连续载荷BER/CRC为null，
不把浮点存储字节当成真实链路比特压缩率，也不声称与数字/JPEG比较完成。

## 上传与安装

把压缩包和.sha256文件上传到服务器项目根目录。IP/SSH别名替换成你的连接：

```bash
scp ~/Desktop/v10_text_recovery_v3.tar.gz ~/Desktop/v10_text_recovery_v3.tar.gz.sha256 zw@服务器IP:/data1/zw/WYH/multimodel/
```

服务器逐段执行，不复制提示符；已有同名文件/输出不删除：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_text_recovery_v3.tar.gz.sha256
test ! -e _v10_text_recovery_v3 && tar -xzf v10_text_recovery_v3.tar.gz
"$PYTHON_BIN" -B _v10_text_recovery_v3/install.py --project-root /data1/zw/WYH/multimodel
```

依赖原V2模块已安装且哈希一致。安装器冲突拒绝，先备份共享HANDOFF再追加；
完全相同的安装可重复核验。没有新依赖或模型下载。服务器现有Python3.8/
Torch1.12/Transformers4.30.2保持不变，读取已有完整bart-base与CLIP预训练。

## 第一轮只运行文字诊断，物理GPU1

```bash
"$PYTHON_BIN" -B v10_extension/text_recovery_v3/launch.py data-check
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used,memory.free --format=csv
"$PYTHON_BIN" -B v10_extension/text_recovery_v3/launch.py diagnostic experiment_results/v10_text_recovery_v3_01 --max-hours 2
```

至少16GiB空闲，运行保留4GiB；不足停止，不杀其他人的进程。整体2小时是
安全上限，不是预计耗时。参考旧V2每组1200步约3-4分钟，新五组＋生成检查
粗估15–45分钟，未经V3实测；日志用真实速度更新阶段ETA。实际显存由smoke
测量，不把旧版本峰值当保证。不会从本地自动启动服务器任务。

SSH会断时只选后台启动一次，不能同时执行上面的前台启动：

```bash
test ! -e experiment_results/v10_text_recovery_v3_01 && test ! -e experiment_results/v10_text_recovery_v3_01_launch.log && nohup "$PYTHON_BIN" -B -u v10_extension/text_recovery_v3/launch.py diagnostic experiment_results/v10_text_recovery_v3_01 --max-hours 2 > experiment_results/v10_text_recovery_v3_01_launch.log 2>&1 &
```

查看进度：

```bash
tail -n 60 experiment_results/v10_text_recovery_v3_01/v10_smoke.log
tail -n 60 experiment_results/v10_text_recovery_v3_01/v10_screen.log
"$PYTHON_BIN" -m json.tool experiment_results/v10_text_recovery_v3_01/v10_screen/progress.json
```

完成后发这个包（负结果/native_stop也发），大权重留服务器：

```text
experiment_results/v10_text_recovery_v3_01/v10_text_recovery_v3_records.tar.gz
```

执行失败没有完成包时，发当前smoke/screen日志及FAILED.json、learning_audit、
bart_interface_audit。需要重跑改02等新目录，保留失败01，不能覆盖重试。
手动收集已完成诊断，不重新训练：

```bash
"$PYTHON_BIN" -B v10_extension/text_recovery_v3/collect.py experiment_results/v10_text_recovery_v3_01/v10_screen v10_text_recovery_v3_records_manual.tar.gz
```

## 后续扩大（先看第一轮结果，不必现在运行）

```bash
"$PYTHON_BIN" -B v10_extension/text_recovery_v3/launch.py pipeline experiment_results/v10_text_recovery_v3_expand_01 --max-hours 3
```

这是新的独立诊断/主训练，会重新做CPU/smoke/五组，不是精确断点续跑。
光配置默认来自experiment_results/v10_joint_jscc_arch_v1_01/v10_screen；若实际
路径不同用`--channel-source-run`指定已完成B4目录。数据、模型、split也可用
明确参数覆盖项目内已有只读输入。不要自动扩大test或恢复到失败后的旧目录。

## 方法来源与验证边界

- BART去噪预训练/自回归生成：[ACL2020原论文](https://aclanthology.org/2020.acl-main.703/)。不是已预训练的模拟信道无损编解码器。
- 标签移位、mask与缓存使用：固定[Transformers4.30.2源码](https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/models/bart/modeling_bart.py)。不更改服务器版本。
- 低秩适配：[LoRA原论文](https://arxiv.org/abs/2106.09685)。手写小型PyTorch层，不安装PEFT；论文没有证明本项目FSO收益。

本地测试明细见[VERIFICATION.md](VERIFICATION.md)。本地无Torch或实际预训练
模型，源解析/NumPy合同/打包还原通过不能当作BART训练有效或论文结论。
