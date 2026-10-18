# V20 首次服务器预检：确定性池化兼容故障

核验日期：2026-10-18。状态：真实失败已定位，**尚无 V20 训练效果**。
本文件只公开运行状态和故障聚合，不含 DOCCI 原始图文、数组、records 或权重。

## 实际执行与未执行

回传 PRIVATE 包 SHA-256：`0f2fa55db55040375eb671b5ee8863499db507f87562f8b03201b0311f51fcae`。
146 普通成员、145 个逐文件哈希通过；20 个原 V20 源码快照和109个依赖逐字节匹配可信版本。
没有在本地重新加载大型缓存或模型权重。

- 服务器完成 TRAIN8/dev2/qual_dev2 的 smoke 缓存，原始独立数组校验回执通过。
- A_self 首次 backward 失败，训练 history 为空，异常早于 optimizer.step，完成更新数为0。
- 其余 smoke、正式8192对缓存、三组长训练和图文评价均 NOT_RUN。
- 起始空闲显存46.90GiB；实际异常不是 CUDA out-of-memory，不能据此预测完整运行峰值。

## 原因与验证遗漏

新 Fusion 的 self 路径使用 adaptive_avg_pool2d；原入口开启严格确定性。
PyTorch1.12.0 官方 CUDA backward 对该算子报告非确定性，因而主动中止：
[版本固定的一手实现](https://github.com/pytorch/pytorch/blob/v1.12.0/aten/src/ATen/native/cuda/AdaptiveAveragePooling.cu#L722-L728)。
原测试使用CPU张量，没有覆盖CUDA+严格确定性+需要梯度的self池化，属于实现验证遗漏。
这不支持“文字辅助无效”或“模型质量失败”的结论。

## Fix1 的范围与验证边界

[修复入口和命令](../../../docci_text_guided_image_v20_fix1/README.md)。
只在原模型模块局部替换4×8池化：16/32/64尺度采用同分区固定均值，保留梯度与严格确定性；
不改全局Torch、不detach、不改参数数量、协议、数据、损失、噪声或验收阈值。
原源码/manifest/失败目录不覆盖。已验证小缓存经独立字节复制复用到新目录；不把12例当正式缓存。

本地15项新CPU测试和61项旧CPU回归通过，源码包8成员独立恢复、Python3.8语法及4段命令语法通过。
3项新CUDA测试及6项旧Torch合同因本机无Torch/HF明确跳过，**不称GPU修复已实测通过**。
服务器修复入口强制先执行CUDA前向/梯度等价、重复一致、三组更新及LPIPS梯度门禁，不许跳过；
之后仍须真实保存父权重/骨干smoke通过才进入原定三组长训练。

源码包 SHA-256：`da4285999c36d1206b6c0dcbb578f9f0e843189b115f414b52309b851d7df51d`。
无需重新下载LPIPS或安装依赖。本会话没有连接服务器或启动远程训练。
