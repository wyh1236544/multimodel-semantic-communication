# V6 fix1：无噪声条件 SNR 核验修复

只替换 V6 的 report.py、test_report.py、MANIFEST.json。模型、训练、信道、
质量门槛及所有旧权重/结果不变。原 V6 压缩包保留，本包须安装在原 V6 上。
根因：clean_args 接收接口传40dB占位条件，结果如实保存40，旧核验器误写20。
40不是实测光学SNR；无噪声对照没有导频测量。FSO仍核对实际导频估计值。

## 上传与安装

将两个下载文件保存到Mac桌面，替换服务器IP：

```bash
scp /Users/a1-6/Desktop/v10_clip_bart_joint_v6_fix1.tar.gz /Users/a1-6/Desktop/v10_clip_bart_joint_v6_fix1.tar.gz.sha256 zw@服务器IP:/data1/zw/WYH/multimodel/
```

服务器逐条执行；任何命令失败先停，不删除冲突：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_clip_bart_joint_v6_fix1.tar.gz.sha256
test ! -e _v10_clip_bart_joint_v6_fix1 && tar -xzf v10_clip_bart_joint_v6_fix1.tar.gz
"$PYTHON_BIN" -B _v10_clip_bart_joint_v6_fix1/install.py --project-root /data1/zw/WYH/multimodel --dry-run
"$PYTHON_BIN" -B _v10_clip_bart_joint_v6_fix1/install.py --project-root /data1/zw/WYH/multimodel
"$PYTHON_BIN" -B v10_extension/clip_bart_joint_v6/install.py --project-root /data1/zw/WYH/multimodel --verify-only
"$PYTHON_BIN" -B v10_extension/clip_bart_joint_v6/launch.py screen experiment_results/v10_clip_bart_joint_v6_02 --max-hours 4
```

自动GPU1、CPU报告/真实Torch接口检查、26项Torch/HF小模型检查、新的真实
saved模型smoke，之后才正式screen。实际 Torch 接口回归不加载预训练权重，
只检查 clean_args 的真实Tensor返回及报告核验。旧失败_01原封不动。
安装前完整哈希检查；原3个源码和HANDOFF备份到deployment_records，
重复安装已验证内容不再写入。未知修改/混合版本拒绝覆盖。
原_package/install.py不重新解压安装；本修复安装器更新后的模块MANIFEST有效。

## 时间、进度、发回结果

smoke5–20分钟、正式screen60–150分钟仍是粗估；当前日志无完整墙钟时间，
不能进一步实测预测。4小时为停止上限。没有精确断点续跑、不用旧smoke权重。
需要独立第二终端看进度：

```bash
tail -n 80 /data1/zw/WYH/multimodel/experiment_results/v10_clip_bart_joint_v6_02/v10_smoke.log
tail -n 80 /data1/zw/WYH/multimodel/experiment_results/v10_clip_bart_joint_v6_02/v10_screen.log
```

完成后发_02/v10_clip_bart_joint_v6_records.tar.gz；异常则先发当前日志，
不删除FAILED，不手改接收记录，不把工程完成当模型收益。

## 验证边界

新增回归先在未修report上真实复现 Received pilot SNR differs，再修最小
核验常数。报告39项：本地38通过，1项真实Torch回归跳过（本地无Torch）；
另10策略、49历史V5/V4回归、8安装合同、实际压缩包还原/安装/重复安装及
原模块verify-only检查。详细实际结果见VERIFICATION.md。
本地未运行GPU、预训练模型、COCO、扩散或服务器任务，必须重跑服务器预检。
