# V22代码交付核验，不是新的模型效果

2026-10-19。用户授权在暂停旧融合扩训后，新增[V22免训练可行性与图像数字基线](../../../docci_pretrained_checks_v22/README.md)。
只有11份源码/文档及MANIFEST共12个tar成员；141份既有依赖固定SHA。

实际本地检查：20项新CPU合同、13项V19旧CPU和16项V21旧CPU回归全部通过。
包含四MCS真正字节/CRC/LDPC/调制无噪声收发、容量边界、16点Gray16QAM、相同AWGN样本与全帧能量。
合成图片的完整JPEG数字阶段、拒包、独立行矩阵、可移植HTML和PRIVATE轻量归档实际执行；这些是虚构测试材料，非DOCCI模型效果。
隔离解压并按141依赖哈希恢复后，20项CPU再次通过。9份Python源码通过3.8语法，11段README命令通过bash语法。
实际inventory子进程只读返回资产状态；没有真实smoke依据的扩大入口拒绝，在创建GPU输出前退出。

源包SHA256：`32d43f698aea3b9a7943c315fc5058fdfe7964dad3ae484495b6390dc7d78f65`。
完整机器交付日志保存在本地deployment_records/v22_QA_20261019_02，不公开PRIVATE结果。
最初未发布候选包及QA保留在01目录；最终包已修正连续/数字生成组的统计隔离与运行版本身份记录。

未运行：真实V20/V18保存权重、BLIP-2、SD、LPIPS新推理、BPG Linux二进制、服务器/GPU显存与新耗时。
本机无Torch/Transformers/Diffusers，不能用CPU合成测试声称模型GPU可用或互促成功。
新模型权重不在代码包中，也未下载；先执行inventory发回ASSETS.json，核实服务器已有目录与版本。
有现有V20资产时可独立做2图prepare，必须重现真实旧权重锚点，再考虑JPEG/新预训练模型预检。

不自动开始任何训练、FSO、模型下载、环境升级或其他用户进程操作。新实验没有语义成功结论。
公共保存仅源码/协议/聚合；真实原始图文、完整records、权重和缓存保持PRIVATE。
