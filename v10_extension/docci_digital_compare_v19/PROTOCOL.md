# V19：文本同资源数字对照与负SNR补测

状态：2026-10-17冻结的探索性协议，已看过V18结果，不是预注册或新测试集。
只新增本模块；不训练、不改变V18权重/结构/载荷/生成设置，不改历史源码或结果。

## 问题与比较

1. 在相同100篇复用qual_dev、每篇9216复符号、单位平均符号能量、同一AWGN噪声实现下，实际数字包与现有连续文本codec的恢复差多少？
2. V18在训练范围0–20dB之外如何退化？这是OOD推理，不称为负SNR训练。
3. Shannon长度预算只作乐观理论参考，不能填成实际正确率、BER或0% WER。

复用V18冻结/clean_continue/noise_replay已保存的clean与20/15/10/5/0dB输出。
新增−10/−8/−6/−5/−4/−2dB，三种权重、每篇每档3次独立噪声。
CPU数字链路测试clean及上述全部11个数值SNR；两个码率均完整报告，不按验证结果挑最优。
固定种子42，噪声调用V18 trial_seed/noise，实维噪声后配对除sqrt(2)构成复噪声。
两方案观察值不同但epsilon相同；SNR指Es/N0，clean=null，不是0dB。

## 实际数字帧（不是完整WLAN PHY）

- UTF-8全文在raw/zlib9/bz2-9/lzma-XZ6中逐消息确定性选最短，平局按此顺序。没有跨描述字典、语言模型或验证集拟合。
- 独立报告四种压缩长度；best_stdlib不称为最强文本压缩器。
- 64个已知单位能量QPSK导频估计噪声功率。h=1、载波/定时理想，与V18相同符号级AWGN假设；没有宣称测过同步或光链路。
- 18字节帧头：4字节magic、1字节版本、1字节压缩ID、4字节原始字节数、4字节压缩字节数、4字节CRC32。
- 帧头144信息bit公开补零到324，IEEE802.11型QC-LDPC N648/R1/2编码，QPSK后发送两份，占648复符号。
- 正文为压缩字节+CRC32，按324或486信息bit补零分块，N648/R1/2或R3/4编码。两个配置的帧头相同。
- 剩余8504复符号全部用于正文：固定公开交织后循环发送码bit，接收LLR按索引相加。重复量只由已编码长度决定，不看SNR或译码结果；无ARQ/反馈。
- 接收端先译码/校验帧头，才获得压缩类型和长度，不能使用发送端包长或原文。公开填充零可用作译码先验，且其资源已计费。
- 自写NumPy系统编码器与归一化min-sum译码器(alpha=.8，最多40轮，syndrome提前停)。采用标准矩阵，不声称译码器达到容量或完整符合WLAN收发标准。
- 帧头/正文CRC、格式、UTF-8或有界解压任一失败，输出空字符串；保留所有失败样本，非空原文对应WER=1。整篇擦除策略可能不如分句/渐进数字方案，必须保留这条限制。
- 所有9216个符号都单位能量，没有免费重复、能量提升或事后噪声归一化。每包输出导频/头/正文/填充/重复开销、实际Es/N0、原始硬判BER、条件于头成功的正文译码BER、包成功率。
- 模拟运行秒数不是传输时间。符号率/净比特率/实际秒级线速/波形/FSO/图像PSNR均未测，本版不填假值。

## 数据、权重和生成

父V14 SHA3e6743eb1bcdbe4bed5db14ba9c9d5cb6d30cb2676066fc41c63fbb09ee3b000。
V18 clean SHA4a0a97034e7b7c2ff48303bfbeadd31677542cccf95ee9ad5e9ba8b02e83f6b1；noise SHA96b27347b62c918a931ba221640139ce3e162b8ff8fbcaebde89f6b13cc5312b。
先用可信V18 CPU验收器复核COMPLETE输入，不重新训练。GPU前逐个核验checkpoint哈希再读取，实际12个锚点生成必须重现V18固定前2篇的clean/0dB输出；加−5dB短smoke后才扩大。
全部BART/codec参数冻结，torch.no_grad；保持贪心、9公开帧、max_length65及原EOS规则。没有Common/CLIP/图像辅助。

## 统计、可视化与结论

主报告每档宏平均规范化WER、完全一致率、包失败率。数字失败也进入分母。
相同描述的3次噪声先平均，再描述级配对bootstrap10000/seed42，点态未校正95%区间；不当作300独立样本，不称多训练种子确认。
先完整画曲线，不预设连续获胜；选出交叉点只能作为下一轮训练设计依据，不能回填新主假设。
原生BART不是等资源数字链路；容量“可容纳比例”不是实测成功率。
事实审核PENDING、BERTScoreNOT_RUN；WER不替代语义准确性。生成30篇固定源事实审核表不自动评分。
真实文本/星座/records均PRIVATE，只允许源码和聚合结论进公开仓库。

## 运行边界

Python3.8/NumPy1.23现有环境即可做数字对照；GPU补测用原Torch1.12/Transformers4.30，无安装下载。
CPU限4线程。GPU物理1，启动至少16GiB空闲、运行保留4GiB；磁盘启动4GiB、运行2GiB。
显式总时限、独立收集预留5分钟；失败/超时保留部分，不自动改batch/重试/杀别人的进程。
digital可以独立CPU运行；negative以新输出只读接续已完成digital，不重复数字主实验。

## 来源（2026-10-17核对）

- [MathWorks LDPC例子](https://www.mathworks.com/help/comm/ref/ldpcencode.html)：N648、Z27、R3/4矩阵及系统编码定义；不依赖MATLAB。
- [Jiri Milos，DREL FEEC BUT，LDPC_matrix.m](https://github.com/jirimilos/802.11ay-phy-sim/blob/main/LDPC_matrix.m)：N648/R1/2矩阵，数值表转录，未复制收发实现；与R3/4官方例子交叉核对约定。
- [Tse与Viswanath，容量章节](https://web.stanford.edu/~dntse/papers/book121004.pdf)：每复符号log2(1+Es/N0)。不适用于未经转换的IM/DD光强容量。
- [Polyanskiy等，有限码长](https://people.lids.mit.edu/yp/homepage/data/finite_block.pdf)：实际有限块长不能用渐近容量直接判成功。
- [DeepJSCC](https://arxiv.org/abs/1809.01733)：参考失配曲线设计，不移用其图像效果。
