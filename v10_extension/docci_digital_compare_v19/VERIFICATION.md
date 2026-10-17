# V19 验证边界

本地CPU可执行：标准矩阵系统编码和独立circulant syndrome、无噪/可纠错比特往返、QPSK理论BER、四压缩器UTF8往返、有界解压、CRC擦除、噪声/资源计数、完整矩阵/篡改拒绝、报告导入与收集、Python3.8语法、隔离源码恢复。

交付前实际执行13项核心+7项存储/篡改/收集合同，及100描述/6800实际数字包的本地CPU端到端联调和全包确定性重放。此处是CPU软件收发证据，不是光硬件、GPU负SNR或模型训练效果。

本地没有Torch/Transformers，不能声称真实BART/保存权重/GPU负SNR推理通过。服务器pipeline/negative强制原V18 Torch合同，再做固定前2篇×3权重×clean/0dB锚点及−5dBsmoke，输出需与V18逐token一致，才扩展。

CPU验收器独立检查源身份、矩阵覆盖、重算WER、噪声种子与方差、符号预算与能量、条件BER分母、冻结回执、容量算术。--replay-digital额外确定性重放实际数字收发代码，确认逐字段结果；不是第二份独立LDPC实现。编码器还以独立circulant校验和非零bit翻转测试交叉验证，但未做MATLAB/WLAN设备互通，不能称标准认证。

清零/失败/空矩阵/重复样本等负向合同必须失败，不能只检查生产者写的passed。原V18轻验收不重新读取未回传的大型数组；GPUItems逐文件核对实际cache SHA，权重先核对固定SHA再反序列化。公开源码包不含任何pickle或原始DOCCI文本。

本版所有比较仍是单训练seed、反复使用的100qual_dev；bootstrap描述不代表训练不确定性。数字包采用全段CRC擦除且只有两种固定LDPC码率，不等同所有数字策略。新负SNR不是模型新训练质量。
