# V14验证边界

2026-10-15，本地macOS独立Python3.12运行，NumPy/Pillow可用；没有Torch/Transformers或GPU。目标服务器Python3.8/Torch1.12/Transformers4.30.2的真实骨干执行必须另行通过。

## 已执行

本轮32项新非Torch合同（core15、report17）与31项指定旧版回归（V10 8、V12 9、V13 14）通过，共63项。5项Torch/HF模型数值合同SKIPPED，不能合计为68项通过。

- `test_core.py`：配置/全9647支持、完整BPE九帧、有效token加权计数、等更新数据量控制、训练尾批、不改旧贪心、SNR不适用、写路径/符号链接、磁盘缓存LRU/校验、离线LPIPS入口和Python3.8语法。
- `test_report.py`：正向校准/训练格式、完整/缺失/重复矩阵、重封哈希后的伪造指标拒绝、源文/错配帧/生成拼接错误、冻结审计、实际训练顺序与有效token计数、失败结果拒绝、网页转义、原始目录只读收集与隔离恢复、事实先定义/盲审/未完成拒绝/unclear留分母。均为明确标注的合成数据，不伪称真实推理。
- 旧V10 `test_core.py` 8项，旧V12 `test_report.py` 9项，旧V13 `test_eval.py` 14项通过；旧模型、评价源码及协议未改。
- 代码包构建器逐文件SHA256、USTAR固定安全成员、隔离解压和文件一致性检查；实际归档身份在随包`.sha256`及发布输出记录中，不把源清单当成归档校验。
- 在独立恢复目录复制42项固定历史依赖，真实执行恢复版core/report32项合同及source_check，均通过；42项依赖的Git blob身份与公开仓库基线46e61d2一致。

## 未执行，不得推断成功

- `test_model.py`的5项真实Torch/HF模型合同：本地因依赖缺失全部SKIPPED，服务器launcher通过V14_REQUIRE_TORCH=1强制执行，缺依赖拒绝继续。
- 官方预训练BART/VAE、批准的真实V12/I2大权重加载、GPU显存峰值、真实DOCCI训练/恢复效果与耗时均未在本地执行。
- `released64`未在本机与整个上游库做张量逐位等价测试，不声称官方完整复现。768容量消融仅为假设。
- 可选LPIPS接口已检查无自动下载与强制权重身份，但真实LPIPS资产/数值前向未执行；默认明确NOT_MEASURED。
- 人工事实标签未填写；没有自动生成“事实准确率”、没有修改历史QUALITY_STOP或成功门槛。
- SwinJSCC、数字FEC/QAM/CRC对照、Common互助、FSO及扩散均不在本包已执行范围。

## 发布与运行安全

仅新增`v10_extension/docci_baseline_v14`；共享导航/状态修改另有原文件备份。没有执行服务器训练、安装依赖或停止外来进程；无学校服务器owned PID。

先校准，再用户显式运行pilot；真正的CUDA数据smoke权重丢弃，main新进程重新初始化。任何工程失败保留现场，不自动换batch、不删FAILED、不自动重试/加训/晋级FSO。固定末次权重，qual_dev反复使用是开发分析，不是独立测试结论。

公开Git只保存源码、方法说明和无原文的聚合结果。运行records中的图文、缓存、完整私有网页、大权重不得公开。
