# 验证范围

本版必须区分源码/合同验证、实际旧记录接入、新评价器推理、人工审核和新训练五层。

本地test_eval.py全部使用明确合成的文本、分数、标签和小资源文件，测试完整矩阵、冻结哈希、拒绝覆盖、源事实先行、330候选盲审、空输出/不确定分母、BERTScore设置和smoke拒绝、可选评分导入及归档成员。它不验证真实CAPTURE/AlignScore、Longformer前向或神经恢复。

交付前还执行：Python3.8语法解析、实际V13/1100条已回传BERTScore的只读适配、实际V14校准文本记录接入、代码包USTAR逐成员SHA和隔离恢复测试。具体执行数与归档身份保存在releases中独立verification.json，不把计划当已通过。

未执行：本地真实Torch/HF/Longformer前向、40条新挑战的真实BERTScore、CAPTURE/AlignScore模型适配或推理、人工事实标注、V14新文本训练、LoRA/InfoNCE/新分段/噪声/数字对照/联合恢复。未安装依赖、未启动学校服务器任务。

源码包不含原始DOCCI图文或权重；真实输出和私有网页只放本地/私有records，不能公开GitHub。公开仅代码、协议、合成测试、聚合结果与边界。
