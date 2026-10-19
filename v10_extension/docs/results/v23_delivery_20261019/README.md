# V23交付核验：免训练数字对照和预训练图像辅助文本

2026-10-19。[源码与服务器命令](../../../docci_evidence_v23/README.md)，
[固定协议](../../../docci_evidence_v23/PROTOCOL.md)。
只新增docci_evidence_v23；旧源码、权重、缓存、数据和结果不覆盖，不训练、不安装环境。
源包13个普通成员，12源码/文档+MANIFEST，153份继承依赖逐文件SHA固定。
代码包42876字节；SHA256 f24e5e5d5f43b3799ff89664f82aa78d67dda5c9578f86d9d62e031323c9bd27。

## 已实际核验

- 28项新CPU合同、20项V22旧CPU回归全部通过；隔离恢复全部153依赖后28项再次通过。
- 10份Python源码按Python3.8语法解析、12段README命令bash语法检查通过。
- 实际子进程inventory、旧V22六case只读导入、24个JPEG四MCS真实收发、独立CPU重放通过。
- 同一组旧PNG的6条LPIPS评分只读导入，不重复GPU评分；新JPEG的LPIPS未运行。
- PRIVATE轻量收集、原始89文件不变、HTML链接/SVG曲线核验通过。
- 无smoke的扩大在GPU加载/输出创建前拒绝；自有子进程超时退出有专门负向测试。
- 下载器默认仅打印计划；range续传、Git blob/LFS验证、空partial与错误响应路径测试通过。
- 官方2023快照API元数据及两份小配置哈希实际核对；未下载15.5GB权重。

本地JPEG使用现有两张保存PNG，只证明代码流程与当前CPU参考结果，不代替服务器环境验证、
32图SNR曲线、强数字基线或任何联合恢复贡献。服务器命令仍先用两图检查本机环境。
新BLIP-2推理、潜变量模型校准、扩大GPU曲线、新JPEG LPIPS及学校服务器新任务均NOT_RUN。
本机无Torch/Transformers/Diffusers，不能用CPU合同声称新神经路径已经成功。

## 三项边界

A：JPEG+CRC+LDPC+QPSK/16QAM，同4128图像复符号/同实际噪声/同能量。
smoke2图×clean/0/−5，quick8/pilot32网格clean/−8/−5/−2/0/2/5/10/20；
pilot3draw是独立单次发送试验，不是重复后合并。JPEG不是最强BPG/JPEG2000数字方案。

B：冻结BLIP-2六组：接收文本、caption-only、联合正确、错配图、灰图、错配文字。
只在clean/0/−5推理。必须独立事实审核，caption-only改善不等于联合恢复。
图像单路4128、加文本系统13376复符号；没有免费原文/原图/真实长度或源mask。
官方2023快照1ef7f63a8f0a144c13fdca8103eb7b4691c74cec避免当前main新配置；
预训练重叠unknown，实际显存/效果待服务器预检，不承诺一定遵循长文本纠错指令。

C：固定TRAIN8/32潜变量误差与同VAE往返校准，保存PNG与服务器NPZ，
独立CPU复核数组统计及PNG指标；不根据dev挑扩散步数、不运行扩散或新训练。
timestep_mapping=null，不能把信道SNR当作SD潜空间噪声强度。

完整QA私有日志：deployment_records/v23_QA_20261019_01/QA.json。
最初未发布候选与本地测试目录保留，不重写原运行版本身份。
公共GitHub只保存代码/协议/聚合/交付状态，无原始图文、records、NPZ或模型权重。
