# V26：诊断V25为何没产生图文互促收益（不训练）

复用已完成`v25_blip_feature_pilot_01`的权重和缓存。只加载BART和两个小适配器，**不加载BLIP-2/VAE、不更新参数、不再缓存图片**。
新增内容TF CE、每段EOS/内容损失、真实有效视觉残差与门值、TRAIN自由生成及160份待人工审核候选。
DEV生成沿用V25已核验结果，不重新生成后替换它们。旧版本及失败结果全部保留。

[固定协议](PROTOCOL.md)。本地只有CPU协议检查，不声称V26真实GPU结果已经得到。

## 1 上传并解压

上传新tar和.sha256到已有multimodel目录。V25及其原有依赖需已安装，不覆盖它们。

```bash
cd /data1/zw/WYH/multimodel &&
sha256sum -c v10_docci_rx_diagnose_v26.tar.gz.sha256 &&
test ! -e v10_extension/docci_rx_diagnose_v26 &&
tar -xzf v10_docci_rx_diagnose_v26.tar.gz --keep-old-files &&
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_rx_diagnose_v26/test_core.py
```

## 2 直接运行只读诊断流水线

```bash
cd /data1/zw/WYH/multimodel
export V26_OUT=/data1/zw/WYH/multimodel/experiment_results/v26_rx_diagnose_01
test ! -e "$V26_OUT" &&
test ! -e "${V26_OUT}_launch.log" &&
nohup /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_rx_diagnose_v26/launch.py pipeline "$V26_OUT" \
  --max-hours 2 > "${V26_OUT}_launch.log" 2>&1 < /dev/null &
```

固定GPU1，启动至少4GiB空闲/运行2GiB余量；只含BART和小适配器，不是V25的BLIP组合。
默认父目录`experiment_results/v25_blip_feature_pilot_01`，BART为`pretrained/bart-base`。
可以用`--parent`/`--bart-path`指定已有资产路径，但字节仍必须匹配固定父权重，不接受别的版本冒充。

顺序：CPU合同→6项真实tiny BART→原V25全量只读核验→小smoke→独立验收→正式只读diagnose→独立验收→PRIVATE收集。
smoke或真实保存权重锚点失败时停止，不继续正式诊断。没有训练入口，不会自动扩训。
预计十几至几十分钟，仅由上轮吞吐估算；2小时是保护上限，不是预计耗时。核验/IO也计入总时间。

```bash
tail -n 80 /data1/zw/WYH/multimodel/experiment_results/v26_rx_diagnose_01_launch.log
tail -n 40 /data1/zw/WYH/multimodel/experiment_results/v26_rx_diagnose_01/diagnose.log
```

后面的diagnose.log未创建时先看启动日志或inputs.log。每条样本输出进度，TRAIN16条新测，DEV32条TF新测但自由生成复用。
TF总1680组不等于1680条独立样本；相同32条DEV反复用过，不能作为论文独立测试。

## 3 完成后发什么

完整完成看`LAUNCH_FINISHED.json`的status=VERIFIED且records存在，不只看PIPELINE COMPLETE。

```bash
cd /data1/zw/WYH/multimodel
test -f experiment_results/v26_rx_diagnose_01/LAUNCH_FINISHED.json &&
test -f experiment_results/v26_rx_diagnose_01_records.tar.gz &&
/home/zw/anaconda3/envs/torch/bin/python -B -c \
"import json; print(json.load(open('experiment_results/v26_rx_diagnose_01/LAUNCH_FINISHED.json')))"
```

发`experiment_results/v26_rx_diagnose_01_records.tar.gz`即可。V25权重/NPZ和所有数据继续留服务器。
包内网页显示TF CE、原/新自由生成WER及残差；FACT_REFERENCES/FACT_CANDIDATES等待人工填写，不自动批准语义成功。
关键区别：TF损失用了真实前缀，只能诊断模型是否读懂特征，不能当成真实接收恢复效果。

失败先发启动日志和对应阶段日志末160行，保留目录，不改依赖或反复启动。
如果只是后续收集需要重做，使用新包名，不训练：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_rx_diagnose_v26/collect.py \
  experiment_results/v26_rx_diagnose_01 \
  experiment_results/v26_rx_diagnose_01_records_collect02.tar.gz
```

可选路径清单（不加载模型、不占GPU）：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_rx_diagnose_v26/launch.py inventory \
  experiment_results/v26_inventory_01 --max-hours 0.5
```

原图文、事实表、PRIVATE网页/records和权重不得上传公开GitHub；只同步源码、协议、聚合状态。
