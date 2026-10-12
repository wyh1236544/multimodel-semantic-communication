# 本地验证与未验证边界

状态：源码候选，未训练/未运行服务器。固定本地bundled Python/NumPy，未安装或升级任何环境。

- `test_core.py`：11项，通过。固定文字预算、文本标注哈希/按图片划分、无图片文件读取、拒绝test源、完整/覆盖指标、WER不截顶、BLEU2、记忆依赖门禁、聚类bootstrap、路径保护。
- `test_channel.py`：6项，通过。48×16公开载荷/384复数拟合、拒绝旧2304载荷/重复、实际NumPy短帧probe+features的完整V3传播与计费、有限输出、失败holdout不能通过。
- `test_report.py`：14项，通过。合成完整与诊断停止矩阵、真实句子指标复算、EOS/源信息旁路/空矩阵/假门禁/假完成、时间能量预算、配对训练初始化、字节校验轻量归档。合成样例不代表模型效果。
- `test_model.py`：8项服务器必需Torch合同，**本地未执行**。缺少Torch会报错，不以skip冒充通过。真实因果AR、source padding、端到端梯度、旧B4文字路径独立与短帧接收等价、权重重载由服务器执行。
- 全部新增Python按Python3.8语法解析。最终源码逐文件哈希、压缩包SHA256、隔离恢复、安装/重复安装/verify、冲突/旧依赖漂移/原字节handoff备份，由build_release.py执行并保存独立JSON。

旧版253依赖清单/字节被保留。GitHub发布前对照远端树逐条核验所需依赖Git blob；共享HANDOFF原字节备份后追加，不覆盖其他远端文件。

实际CLIP/B4权重载入、真实Native/AR梯度和恢复、FSO短bank门禁、GPU显存/耗时、WER改善、真实数字基线、真实光硬件均未在本机验证。源码/NumPy测试通过不等于训练模型有效。服务器pipeline强制先跑Torch合同和真实短预检，质量/代理门禁失败保留证据并停止。
