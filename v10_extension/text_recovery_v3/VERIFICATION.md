# Text recovery V3 验证范围

状态：本地非Torch验证与源发布；实际Torch/HF/GPU由服务器门禁验证。

本地使用任务附带Python3.12与已有NumPy；不安装Torch、不改任何环境。

实际运行后记录于releases/v10_text_recovery_v3_build_verification.json：
Python3.8语法检查、源包逐文件SHA256还原、增量安装/重复安装、旧依赖漂移拒绝、
同名用户文件冲突拒绝、符号链接拒绝、HANDOFF精确字节备份及历史源码不变。

本地test_core.py共16项：配置/七位帧头所有合法长度/损伤不被真值修正/形状
与非有限值/帧头计费/CE渐变/旧质量门槛与新增EOS门槛/独立文本指标重算/
重复缺失划分拒绝/四种真实紧凑帧完整NumPy光波形的资源与事件核验。
本地不执行test_model.py，也没有在本地证明真实BART恢复质量。

本地test_report.py另外7项：负结果合法但非质量成功、五组矩阵、指标/训练
步数篡改拒绝、封存后哈希变更拒绝、执行失败禁止收集、轻量包还原与重验。
共23项新非Torch测试；这些测试中的诊断记录是人工合成fixture，不是模型结果。

另实际执行既有V2 caption/compact测试11项和V3光链路core/dsp/repairs共42项，
合计53项继承回归均exit0。包括完整的模拟星座/帧CRC数字合同，但并非本轮
训练模型与数字方案的新效果对比。

服务器check必须实际执行test_model.py的10项Torch/HF小模型合同：LoRA初始
恒等及冻结、同信道长度边界、源PAD不可泄漏、功率和符号预算、整句/逐步
logits一致性、label PAD忽略、必需组梯度与更新、LoRA重置、失锁清零、状态
重载。随机小模型成功不等于真实预训练GPU训练成功；后面smoke仍必须运行。

默认diagnostic只包含原生诊断与五个干净TRAIN过拟合对照；质量失败保存而
不是COMPLETE。可选pipeline重置后才扩大训练/独立validation/全FSO波形。
全部运行身份、每步日志、实际输出、学习/冻结/重载审计、质量门槛、源与
权重哈希纳入轻量记录包。权重不随包上传，离线检查不能重新执行神经网络。

原始项目没有本地Git工作树，因此GitHub同步使用已连接官方connector建立
单个增量tree/commit并fast-forward主分支；不重建整棵树，不删除旧文件，
先验证旧HANDOFF的blob身份，再同步新源码与准确字节备份。同步状态另记录
在releases/v10_text_recovery_v3_github_sync.json，未成功不得声称已同步。
