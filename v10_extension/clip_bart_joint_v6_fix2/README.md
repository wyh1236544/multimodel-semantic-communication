# V6 fix2：图文联合摘要与文本摘要分开核验

本包安装在已打fix1的V6上。修复结果元数据与核验，不改变模型、训练、信道、
恢复方式、质量门槛或资源开销。保留fix1的clean40dB占位修复。

原source_sha256仍为SHA256(图像float32字节+规范化原文)，绝不改成纯文本。
新增text_source_sha256仅用于核验；同时与此前独立BART文本清单比对图像ID、
选定标注ID、标注文件SHA、划分和UTF8长度。接收端不读取这些源码元数据。
生成适配器训练原文缓存继续TRAIN-only，必须匹配两个独立文本清单。

## 上传、解压、安装、运行

下载tar.gz和.sha256到Mac桌面，按原方式上传到服务器项目根目录，或替换IP：

```bash
scp /Users/a1-6/Desktop/v10_clip_bart_joint_v6_fix2.tar.gz /Users/a1-6/Desktop/v10_clip_bart_joint_v6_fix2.tar.gz.sha256 zw@服务器IP:/data1/zw/WYH/multimodel/
```

服务器逐条执行，任何命令失败先停：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_clip_bart_joint_v6_fix2.tar.gz.sha256
test ! -e _v10_clip_bart_joint_v6_fix2 && tar -xzf v10_clip_bart_joint_v6_fix2.tar.gz
"$PYTHON_BIN" -B _v10_clip_bart_joint_v6_fix2/install.py --project-root /data1/zw/WYH/multimodel --dry-run
"$PYTHON_BIN" -B _v10_clip_bart_joint_v6_fix2/install.py --project-root /data1/zw/WYH/multimodel
"$PYTHON_BIN" -B v10_extension/clip_bart_joint_v6/install.py --project-root /data1/zw/WYH/multimodel --verify-only
"$PYTHON_BIN" -B v10_extension/clip_bart_joint_v6/launch.py screen experiment_results/v10_clip_bart_joint_v6_03 --max-hours 4
```

自动GPU1，先报告/策略/真实Torch接口及原26Torch/HF检查，再新真实smoke，
通过才正式screen。保留失败_01/_02，不删除FAILED、不重写旧清单/NPZ。
安装器严格核对已发布fix1原源码、未改V6源码和378历史依赖，再备份六个
源码及HANDOFF后替换；拒绝未知修改/混合版本/符号链接，支持幂等重复安装。
无需再解压原V6或fix1包、无需下载权重和更换环境。

预计预检5–20分钟、正式screen60–150分钟仍是粗估，当前日志不能证明完整
耗时；4h为停止上限。重新从原正式V5文本父权重预检，不用失败smoke权重。
第二终端查看：

```bash
tail -n 80 /data1/zw/WYH/multimodel/experiment_results/v10_clip_bart_joint_v6_03/v10_smoke.log
tail -n 80 /data1/zw/WYH/multimodel/experiment_results/v10_clip_bart_joint_v6_03/v10_screen.log
```

完成后发_03/v10_clip_bart_joint_v6_records.tar.gz；异常先发当前log。
本地无Torch/COCO/预训练，元数据和合成检查通过不等于GPU预检或收益已证明。
验证范围见VERIFICATION.md；代码已要求实际缓存和独立文本源先交叉核验，
不能为了通过删掉检查，也不把工程完成标记当恢复效果成功。
