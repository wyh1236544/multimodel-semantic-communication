# V13 离线评价交付与核验

V13是评价工作流，不是新训练模型。[完整运行步骤](../../../docci_semantic_eval_v13/README.md)。

- 源码包：`v10_docci_semantic_eval_v13.tar.gz`，22798字节，10个文件。
- SHA-256：`d803aa7e164de1c94d21cd60caba1064be8be3b5fbe534780b64c94bd5c2d0fb`。
- Python3.8语法解析、14项新CPU合成合同、打包后隔离恢复与重跑均通过。
- 原V12的16项CPU回归通过，66个历史V10/V11/V12模块文件与发布前GitHub字节一致。
- 本地真实records适配：100条描述、1100个输出，完整矩阵及逐文本WER重算通过。
- 固定30条长度分层描述供人工审核，覆盖330个候选输出。
- **真实BERTScore模型推理NOT_EXECUTED、人工事实审核PENDING；数字基线NOT_RUN；旧QUALITY_STOP不变。**
- 不安装或修改服务器环境，不启动训练；可选语义模型依赖仅在用户本地独立虚拟环境。
- 旧共享文档在本地发布前备份；Git历史和HANDOFF保留旧状态。
- 包仅包含新增模块。GitHub另同步当前状态和V12聚合结果，不包含原始图文/完整records/权重。

[机器可读核验收据](verification.json)。源码包内部的MANIFEST可逐文件核对本次Git源码。
