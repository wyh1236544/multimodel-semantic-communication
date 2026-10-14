# DOCCI文本纯CE权重续训 V11

状态：新增实现，真实预训练BART/GPU续训效果未测。只推进文本，不改网络、旧gate、图像或FSO。保留所有历史模块、结果和权重，使用现有离线Python3.8/Torch1.12环境、物理GPU1。

## 实际改动

- 从明确认证的这轮2048篇/batch2/10000步权重开始，再训练10480步，累计20480步=20轮。新AdamW、新阶段LR3e-5线性降至1e-5；不是恢复旧优化器，也不是把旧学习率追溯改为20轮计划。
- 数据顺序沿用全局步数10000，不从第一条重新采样；原TRAIN2048/qual_dev100、BART、codec、9216连续复符号/R1均不变。全部9个公开段，包括空段，继续计费和解码。
- 每2000个追加步记录固定TRAIN8/VAL8自由生成、WER、短停/上限/重复与真实前缀诊断。最终评估TRAIN32/VAL100，保留全部文本及正常/清零/打乱对照；最终权重固定，不按验证集选最好。
- 真实前缀/原生BART记忆仅在离线诊断中用，不进入真实接收端或训练损失。记录gold-prefix内容CE、准确率、EOS概率，以及相同实际可用位置上的free-prefix CE；这是位置对齐的诊断，不是暴露偏差唯一根因证明。
- 满段62正文token后EOS不算提前结束；按长度短停、上限停止和重复分别报告。长度正确也不等于内容正确；源长度只供评价，未传给RX。
- 精确整数位向量编辑距离加快WER/CER，保持旧分词/规范化/BLEU定义；只校验一次，没有旧180秒硬限制。评价和校验有进度。
- 新检查点保存权重、AdamW与RNG状态，但本次入口只接受认证的旧V10权重；失败后不自动精确恢复、不自动重试。

## 上传、校验、解压、安装

把源码包及sha256文件上传到`/data1/zw/WYH/multimodel`，不需要重新上传DOCCI或BART。

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_docci_text_continue_v11_fix1.tar.gz.sha256
test ! -e _v10_docci_text_continue_v11 && tar -xzf v10_docci_text_continue_v11_fix1.tar.gz --keep-old-files
"$PYTHON_BIN" -B _v10_docci_text_continue_v11/install.py --project-root .
```

安装器检查所有新目标和历史依赖，先备份旧HANDOFF字节再追加。若已安装同一源码，重复安装不改写；未知改动/符号链接/冲突拒绝。不要删除旧暂存、代码或结果。不同新版使用不同模块，不覆盖未发布的其他草稿。

## 开始训练（先看GPU空闲）

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
export CUDA_VISIBLE_DEVICES=1
nvidia-smi -i 1 --query-gpu=name,memory.used,memory.free --format=csv
"$PYTHON_BIN" -B -u v10_extension/docci_text_continue_v11/launch.py pipeline \
  experiment_results/v10_docci_text_continue_v11_01 --max-hours 4
```

先CPU合同与实际Torch/HF测试，再真实认证检查点/BART/GPU反向及磁盘回读smoke；恢复原父权重后才做主阶段。默认父目录是
`experiment_results/v10_text_ce_2048_20261014_171836/screen_T1_chunks`，权重SHA256
`72cf3f355e3bd9f84fe8d6d9c8b3e454ed17a05f1a637eab531743794fe993d9`。
必须保留原完整记录和`train_clean/last.pth`。源码包不含权重，不会退回V6或使用其他同名检查点。更换文件位置可以用`--parent-run`指定项目内同字节目录；不能更换不同权重。

若只想先做真实smoke：`launch.py smoke experiment_results/v10_docci_text_continue_v11_smoke_01 --max-hours 1`。随后完整pipeline仍有自己的预检，不能用smoke权重启动。

## 后台启动与进度

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
export CUDA_VISIBLE_DEVICES=1
nohup "$PYTHON_BIN" -B -u v10_extension/docci_text_continue_v11/launch.py pipeline \
  experiment_results/v10_docci_text_continue_v11_01 --max-hours 4 \
  > experiment_results/v10_docci_text_continue_v11_01_launch.log 2>&1 < /dev/null &
echo "启动PID：$!"
```

上面的前台/后台二选一，不要都启动。若目录已存在，保留并用`_02`等新路径；旧失败不能通过删除标记变为完成。输出父目录experiment_results已由旧实验创建。

```bash
tail -n 40 experiment_results/v10_docci_text_continue_v11_01_launch.log
tail -f experiment_results/v10_docci_text_continue_v11_01_run.log
```

`global=10001/20480`表示主续训开始；中途EVAL/PREFIX不是卡住。Ctrl+C退出tail不会停后台训练。实际训练循环按这轮旧速度估计约16分钟，探针、100条评价、读写另外耗时；整轮暂规划30–70分钟，未实测。4小时是停止预算，不是ETA。显存峰值不能从旧2.08GiB保证，运行要求空闲16GiB并保留4GiB，不停止他人进程。

## 完成与结果发回

训练进程结束显示`TRAIN_AND_EVAL_FINISHED`，还不是整体完成；后续独立CPU验收和打包成功才显示`Send ...`。
最终`COMPLETE.json`只表示工程记录完成，`summary.json`同时保留旧QUALITY_STOP/CLEAN_READY和原来的WER/margin/全EOS判断，没有悄悄放宽阈值。

发回`experiment_results/v10_docci_text_continue_v11_01/v10_docci_text_continue_v11_records.tar.gz`；权重、优化器/RNG、原始数据与缓存留服务器。HTML包含全部100条原文/父权重/末权重对比，诊断不伪装成真实恢复。

只有CPU收集失败时，可以在不重训的情况下显式运行：

```bash
"$PYTHON_BIN" -B -u v10_extension/docci_text_continue_v11/collect.py \
  experiment_results/v10_docci_text_continue_v11_01 \
  experiment_results/v10_docci_text_continue_v11_01/v10_docci_text_continue_v11_records_collect02.tar.gz
```

工程错误发`_run.log`末100行和`FAILED.json`，不要重新启动已有输出。新源码没有scheduled sampling、蒸馏或unlikelihood；根据本次gold/free-prefix诊断再决定下一组等资源对照。没有图文联合、FSO抗噪或数字基线增益的结论。
