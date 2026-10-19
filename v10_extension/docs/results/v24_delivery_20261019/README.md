# V24交付：BLIP-2基础看图校准，不训练

状态：2026-10-19代码/CPU交付验证完成；真实tinyHF、完整模型/GPU、新效果仍NOT_EXECUTED。
没有启动学校服务器进程。保留全部旧源码/manifest/权重/数据/结果；共享文档修改前已作字节备份。

[操作说明](../../../docci_caption_calibration_v24/README.md) · [固定协议](../../../docci_caption_calibration_v24/PROTOCOL.md)

直接新增`v10_extension/docci_caption_calibration_v24`，12个源码包成员，166个旧依赖身份固定。
复用已封存BLIP-2及V23两图prepared，20行基础caption，通常9次实际调用。
省略text、64新token、greedy；原图仅特权诊断，真实接收/错图/灰图明确分开。
完整token/EOS、实际前向有限性、逐步概率、模型state前后SHA、显存峰值与PRIVATE网页。
QA是后续显式阶段，需同一caption结果四项人工可用性审核；没有自动扩大/训练入口。

本地28新CPU +28旧V23 CPU回归通过；隔离恢复后28新CPU再次通过。
9个Python文件按3.8语法检查、7段README命令语法、12个USTAR成员哈希通过。
真实已回传PNG只读导入20行/9个图片身份、原文件未改。合成神经替身仅验证生产编排/收集路径，
不冒充神经效果；本机没有Torch/HF，2项真实tinyHF合同必须在服务器通过后才能加载完整模型。
PRIVATE QA位于`deployment_records/v24_QA_20261019_01/QA.json`，不将其模拟输出发布为模型结果。

源码包SHA256：`660cc3f0b276fb336ce36d03ec7f50c65ab46818e5d50f2950c1986be21fb4a1`。
默认GPU1、FP16起始20GiB空闲/运行4GiB余量；30分钟为硬停止预算，非ETA/峰值保证。
结束发送`v24_caption_calibration_01_records.tar.gz`，不上传权重；真实事实可用性人工待审。
