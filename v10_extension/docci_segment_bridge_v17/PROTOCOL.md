# V17 冻结协议：按段视觉修复与文本符号噪声补偿

日期2026-10-16。状态：新实现，真实GPU训练/耗时/质量未执行。用户明确授权隔夜四组有限实验，不授权停止其他作业、改环境或自动进入实际图像/FSO。旧V16数值门槛和失败结果不改。

## 研究问题与借鉴边界

V16正确oracle WER35.9405%，同规模self36.6213%，错配35.5306%；未证明内容相关帮助。新的可证伪问题：在相同自身文本观测下，按段查询视觉区域、加入明确内容对齐和配对匹配监督，是否能让正确图像优于同规模self及多个真实错配？在文本符号受损时是否更明显？

[BLIP-2 §3.2–3.3](https://proceedings.mlr.press/v202/li23q/li23q.pdf)启发：冻结主干、先表示对齐再生成，并结合对比/匹配/生成目标。[ALBEF §3.2](https://proceedings.neurips.cc/paper/2021/file/505259756244493872b7709a8a01b536-Paper.pdf)启发：对齐与跨模态交互联合监督。这里是独立小型接收模块，不下载其模型/权重、不声称复现论文或其大规模数据收益；负例是预声明真实循环/确定性不同TRAIN样本，不声称实现论文hard-negative mining。

## 不变量

- 复用完整V16缓存index SHA256 `ea78989f97a3db56b21510324ee2555245e59c9e8a346bb675cac3ce2a083d23`。
- 原V14 released64文本权重与expanded-I2图像权重的SHA沿用原批准值，所有旧源码/manifest/结果/缓存不写入。
- 9647官方TRAIN、100已反复使用qual_dev，不读test/qual_test；seed42、固定末次权重。
- 18432文本实数＋8192图像实数＝13312复符号计数，R1，无新Common、波形、实际速率/BER或数字基线。
- oracle读原图冻结Long-CLIP patch，是未计成本的理想辅助，非等资源通信系统。图像通路固定理想，只有文本符号加噪，是非对称诊断，不是整条FSO链路。
- 所有原codec/BART/VAE/Long-CLIP权重冻结；运行中仅加载原文本codec/BART，图像特征已缓存。两种桥的训练参数完全一致，完整计算量不宣称相等。

## 网络与教师

每段9×32×768接收memory经归一化/256维投影，加公开固定段位置，用两个交叉注意力块查询64×256辅助槽。段查询来自实际接收文本，不来自原文。带匹配置信门、接收EOS内容门和零初始化有界残差，修复原memory，再交给未修改的BART。关闭修复严格返回原memory，不插BART hook、不改其KV缓存。

辅助输入统一196×768：oracle是原图patch，self是自己同一次观测的memory的无参数固定池化。self不能读取干净memory或图像。新桥的独立参数公式为52w²＋5dw＋(slots＋67)w＋7d＋2；默认4,429,826，真实Torch运行必须另行匹配。语义监督头仅TRAIN使用，匹配头进入接收置信门，两组均含同样的头。

TRAIN-only教师由冻结原生BART encoder根据源token产生，各段内容token上下文均值和按内容token数加权的全文均值，不再将收到memory的随机投影当唯一语义目标。教师和真实内容计数只作TRAIN损失；validation不生成教师，接收接口无原文/源mask/真实长度/段数/错误位置。空段诊断用源长度仅在评价时计算。

## 信道与共同随机数

对原单位均功率18432实数TX wire施加 `y=x+n, n~N(0,10^(-SNR/10))`，不在加噪后重新归一化、不把噪声直接加在clean memory上。再经过冻结原接收codec，重新从观测memory计算第一自由EOS概率门。缓存的clean门从不用于受损接收。

clean：严格恒等，SNR字段null；不是0dB或40dB。训练混合从clean/20/10/5/0dB按phase、epoch、example ID确定性抽取，同一对C/D使用同一条件与噪声实现，各epoch刷新；无SNR真值或教师作为RX条件。报告名义SNR、实测符号功率/噪声功率/比值SNR、wire/memory/gate身份。使用确定性Torch算法和子进程内CUBLAS设置，不改全局环境。

正式自由生成评价预先固定clean/10/0dB，全部100条；训练中的20/5dB不另称已完成自由生成评价。验证每个条件每条1个固定噪声实现，不能把5个donor当额外独立样本，未覆盖多次信道波动。

## 四组与训练

| 组 | 辅助 | TRAIN信道 |
|---|---|---|
| A_self_clean | 自身接收memory | clean |
| B_oracle_clean | 原图特征 | clean |
| C_self_mixed | 自身受损接收memory | clean/20/10/5/0dB |
| D_oracle_mixed | 原图特征 | 与C完全相同 |

主训练顺序固定C、D、A、B，优先完成带噪主问题。不是按结果挑优先级。
每组真实batch64对齐2轮（302更新）；CE micro1/累积8、6轮（7236更新，57882篇呈现）。同对数据次序、own observation和噪声身份严格校验。

对齐loss＝对称InfoNCE＋0.5段教师余弦损失＋0.2匹配BCE。CE阶段＝按有效target token加权CE＋0.05段教师损失＋0.1匹配BCE。匹配负例来自不同TRAIN pair，CE只监督正确配对恢复，不最大化错配生成错误。梯度累积8不是batch8对比学习；InfoNCE只在真正batch64的对齐阶段计算。

小smoke独立从同一完整缓存取8TRAIN/2VAL：每组2次对齐、5次CE、clean和0dB评价，全部4组必须真实完成、冻结/梯度/保存回读和共同噪声检验通过，才启动主阶段。smoke质量波动不当成主实验收益，smoke桥不用于主初始化。

## 对照、评价和成功条件

自己的受损文本固定。每个条件评价正确辅助、固定真实donor偏移1/17/43/61/83、零辅助、关闭旁路。所有donor非自身且互不重复。off重用在准备阶段独立真实生成的冻结基线，明确记录复用，并在每组检查disabled memory逐bit恒等；不是声称每组又重跑off神经生成。

主比较预先指定D−C在0dB；次比较B−A在clean；其余SNR探索性。主指标沿用逐描述normalized WER平均，另报corpus WER、重复、提前EOS、容量停止、空段误激活及真实样例。所有旧评价/QUALITY_STOP不改。容量停止不自动等于循环，EOS也不自动算提前结束。

数值筛查：正确oracle相对同规模self WER至少降低2pp，配对95%区间下限>0；相对5个donor的描述内平均WER至少降低1pp且区间下限>0；首段token和全文word相邻重复不增，空段误激活0。10000次描述级配对bootstrap、seed42，点态未多重校正，不涵盖训练随机性。两条比较完整报告，不事后换主指标/挑checkpoint。

即使数值过关，独立事实审核仍PENDING、full_semantic_success仍false。BERTScore/CAPTURE/AlignScore未在本版实现或伪填。Long-CLIP参与模型，不用它给自身恢复做唯一评价。不得宣称双向互促、连续优于数字或FSO抗噪成功。

## 时间、状态与恢复边界

主入口最多8小时，包含校验/真实smoke/教师与观测准备/四组训练/评价/收集，预留15分钟收集。6–8小时只是排期，不是新架构实测ETA；memory修复需通过更多冻结BART层回传梯度，与V16不同，可能超预算。各组日志ETA仅该组CE，不含剩余组和评价。

启动GPU1可用至少16GiB、运行全卡保留4GiB；启动项目磁盘至少64GiB、运行保留4GiB，均是保护阈值非已测峰值。CPU线程4，有界LRU，teacher约300MB稠密理论CPU数组、验证observations约300MB，不重建原大缓存。

工程异常/非有限数值/资源保护/时间上限停止本流水，保留已完成和部分产物，不自动重试、不降低batch/轮数、不杀其他进程。每epoch保存恢复材料，但未实现精确断点续训，不能把partial epoch当正式last。独立完成组仍可验证与收集，未完成/NOT_RUN明确保留。质量未达筛查不自动发起更多训练，也不当软件故障重复跑。

完整成功状态仅四组全验收后的COMPLETE；PARTIAL_BUDGET/FAILED/SMOKE_COMPLETE都不是完整主实验。只要已完成C/D，可以报告这对的受限结果，但不能把A/B写成已运行。不自动进入actual或FSO。
