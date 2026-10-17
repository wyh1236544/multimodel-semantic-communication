# V20 文字辅助图像：编码＋空间条件解码

先读[实验协议](PROTOCOL.md)。本版第一阶段是**文字→图像**，不是已经完成双向互促。
不重复训练V14/V18文本；图像codec TX/RX和空间融合共同训练。
8192训练对、每组8轮、三组等预算，VAE/BART/Long-CLIP/V18文本冻结。

## 上传与解压

上传到现有 /data1/zw/WYH/multimodel/：

- v10_docci_text_guided_image_v20.tar.gz 及 .sha256（源码）
- v20_lpips_offline_assets.tar.gz 及 .sha256（约233MiB未压缩官方权重；不安装依赖）

在服务器逐行执行：

~~~bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_docci_text_guided_image_v20.tar.gz.sha256 &&
test ! -e v10_extension/docci_text_guided_image_v20 &&
tar -xzf v10_docci_text_guided_image_v20.tar.gz --keep-old-files

sha256sum -c v20_lpips_offline_assets.tar.gz.sha256 &&
test ! -e _v20_lpips_assets &&
tar -xzf v20_lpips_offline_assets.tar.gz --keep-old-files
~~~

任何一步失败停在该步。不要手改旧代码/删除目录重试。已存在相同材料不需重复解压；
入口再次独立检查所有源码、依赖及两个官方权重整文件SHA。
DOCCI仍从项目根读取 docci_descriptions.jsonlines 与已经解压的图片，
没有新下载、COCO回退或服务器环境变化。

默认需要这些现有父结果（权重和父训练集合都校验已知SHA）：

~~~text
experiment_results/v14_image_pilot_02/main/image/expanded/train/last.pth
experiment_results/v18_text_noise_01/train_noise_replay/last.pth
experiment_results/v14_image_pilot_02/main/image/cache_train_identity.json
experiment_results/v14_text_pilot_01/main/text/cache_train_identity.json
~~~

## 一次启动：合同→真实smoke→三组长训练→评价→自动收集

无需分别再跑多个训练命令。**只执行一次下面的pipeline**，不会和文本旧任务重复。
使用物理GPU1（进程内cuda:0），起始空闲至少20GiB，运行保留4GiB；
项目空闲磁盘至少64GiB。不停止别人的任务。

~~~bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
export V20_OUT=experiment_results/v20_text_guided_image_01
test ! -e "$V20_OUT" &&
test ! -e "$V20_OUT"_launch.log &&
nohup "$PYTHON_BIN" -B -u \
  v10_extension/docci_text_guided_image_v20/launch.py pipeline "$V20_OUT" \
  --max-hours 48 > "$V20_OUT"_launch.log 2>&1 < /dev/null &
~~~

48小时是包含收集的硬上限，不是预测。真实saved-parent smoke三组各2次update、
真实VAE/Long-CLIP/BART/LPIPS及磁盘checkpoint回读通过，才自动进入主阶段。
主阶段顺序B、C、A，优先完成接收互助主对照；每组16384次update，
不按验证结果早停/选最佳权重。每轮epoch另存权重，固定末次用于正式评价。
超时/显存/数值/验收失败保留现场，自动轻量收集已完成部分；不自动重试或改batch。
epoch权重备份不是已经实现的精确断点续训。

按旧V14速度外推，新三组总图次约旧单组6.4倍×3，并新增空间注意力和LPIPS。
**整套暂估20–40小时，未经新GPU实测，可能更久**。先看smoke实测速率；
主训练每20更新输出该组ETA，不能把该组ETA当全部剩余时间。
缓存/全图像字节审计、两套评价、生成文字与CPU收集单独耗时。

只检查代码/已有环境而不开始GPU任务：

~~~bash
"$PYTHON_BIN" -B v10_extension/docci_text_guided_image_v20/launch.py check
~~~

## 看进度与结果

~~~bash
tail -n 40 "$V20_OUT"_launch.log
tail -f "$V20_OUT/train_B_tx_text.log"
~~~

B训练后查看 eval_B_tx_text.log、train_C_tx_rx_text.log、
eval_C_tx_rx_text.log、train_A_self.log。缓存阶段看 cache.log，
真实预检看 smoke_cache.log / smoke_train_A_self.log 等。
如果换了终端，先重新设置上面两项export。

结束会产生：

~~~text
experiment_results/v20_text_guided_image_01/PIPELINE.json
experiment_results/v20_text_guided_image_01/REPORT.json
experiment_results/v20_text_guided_image_01/index.html          PRIVATE图文展示
experiment_results/v20_text_guided_image_01_records.tar.gz
experiment_results/v20_text_guided_image_01_records.tar.gz.sha256
~~~

把最后两项发回来，不下载大cache/权重。若自动收集失败：

~~~bash
"$PYTHON_BIN" -B -u v10_extension/docci_text_guided_image_v20/collect.py \
  experiment_results/v20_text_guided_image_01 \
  experiment_results/v20_text_guided_image_01_records_collect02.tar.gz \
  --max-minutes 60
~~~

它只做独立CPU验收/收集，不重新训练。失败时同时发对应阶段日志末尾。
只上传代码/协议/聚合到公共GitHub；DOCCI文本/图片/PRIVATE网页/records/权重保持私有。
