# V12 验证范围

本机没有Torch/Transformers/Diffusers；六项真实Torch/HF合同明确SKIPPED，真实批准权重、512反向、DOCCI、GPU速度与质量均NOT_EXECUTED。
服务器launch.py强制V12_REQUIRE_TORCH=1，缺库即停止，不把跳过当通过，不安装依赖。
本地已执行纯NumPy/stdlib合同、正反向记录验收与历史V11真实存储帧回归；它们不是新模型推理。
确定性源码归档、逐文件恢复、安装/重装/冲突/漂移/符号链接/精确HANDOFF备份等证据见releases中同名build_verification.json。

验收分层：
- CPU合同与安装：路径/源码完整性、资源口径、loss与采样协议、完整矩阵、反例必须拒绝。
- 服务器真实smoke：重载批准V11/I2+BART/VAE/Long-CLIP；实际反向、冻结审计、检查点磁盘往返，独立CPU验收通过后才开始主训练。
- 主实验：固定初始权重/数据/顺序/预算；最终质量由数据决定。文本旧门槛不改；图像只是ORACLE诊断信号，不自动进入FSO。
- 收集：原结果只读，新归档和新派生报告；不把工程完成标记当科学成功。

完整用户操作与剩余限制见README.md。没有在服务器启动任务，没有改全局环境、根目录旧代码、数据/权重或已有结果。
