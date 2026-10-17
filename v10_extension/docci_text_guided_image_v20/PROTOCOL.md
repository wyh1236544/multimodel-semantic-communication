# V20：图像主导的文字辅助连续编解码

2026-10-17；实现方案，不是新实验结果。第一阶段只测文字→图像。
反向图像→文本、Common、扩散、FSO及数字图像对照不在本轮自动运行范围。

## 假设与区别

V12文字接收适配、V16/V17视觉文本接收桥均未证明内容特异收益。
本版从V14 expanded I2图像和V18 noise_replay文本出发：
图像codec TX/RX与四个空间融合模块一起训练；VAE、BART、
Long-CLIP与V18文本codec冻结。不是再次只训练末端小桥，也不是全解冻。

三组同初始参数、同参数数目、相同optimizer/schedule/训练对/信道噪声：
A_self四模块均读取自身图像特征；B_tx_text发送端读取真实文字Long-CLIP特征，
接收端仍self；C_tx_rx_text另读取实际有噪文本符号经V18重建的memory。
四个融合点：编码32/16空间网格、解码32/64网格。32个学习槽重采样，
空间query cross-attention，有界小残差，无内容匹配开关。
同参数不等于同全部FLOPs：文字Long-CLIP源特征固定预计算，
C的接收侧KV长度也更大；这些额外计算成本单列，不声称完全等计算量。

TX Long-CLIP对完整描述分246-content块编码，再固定池化32槽；不截断。
RX只读取9×32×768的实际文本memory；没有原始文字/源mask/长度/矩形、
原图视觉skip或未计费的source辅助。冻结文本wire可缓存，图像wire不可缓存。
这不是TACO、DeepJSCC-WZ或M3AE的论文完整复现；没有加入未实施的损失。

## 数据与预训练暴露范围

原DOCCI official TRAIN9647/qual_dev100。V14两个父权重TRAIN2048元数据
各有已核验SHA，检查真实图像字节哈希及完整文本哈希。
从未见于这些已知父cohort的TRAIN候选按公开seed哈希固定500条，全部排除新训练，
其中前100作为dev主评价。新训练取其余中的8192对，8轮/每组65536图次，
microbatch1累积4/每组16384更新。qual_dev100仅历史复用探索性参照。
禁止读取官方test/qual_test样本内容用于训练、校准或选模。

旧V16/V17已用全9647训练，但其桥/权重/teacher均不复用；
新dev仅称“未用于已知V14/V18父训练和本轮新训练”，不声称从未进入
任何通用预训练集或任何历史研究决策。实际byte/text跨集合重复则停止，不删除难样本。
不按dev选checkpoint，固定末次权重；epoch备份不是自动精确resume。

## 训练与物理资源

单张完整512letterbox，原矩形仅loss/评价。无随机crop/flip。
pixel_MSE + 0.01 latent_MSE + 0.05 LPIPS-Alex-v0.1，
三组完全相同；冻结VAE仍向其输入传递梯度。没有CLIP自评分训练目标。
25%期望clean，其余0/5/10/15/20dB，按epoch/ID固定抽样，不按arm变化。
图文两路同时经历同名义SNR、独立AWGN；无完美辅助，clean单列null。
原图像8192实数＋文本18432实数＝13312复载荷/R1。
每模态32复导频，总64复导频，合计13376复使用；空文本槽计费。
单位功率；复映射(x1+j*x2)/sqrt2，无有噪后归一化。
接收图像条件只用导频估计噪声/可靠性，40仅clean哨兵，非实测SNR。
本版unit-gain AWGN，非FSO；不报告Gb/s/真实光功率/BER。

## 评价与预先固定的数值筛查

dev100为主，历史qual_dev100单列。clean1次、10/5/0dB各3次，
均为相同内容独立噪声试验，不是3次重复发送/合并，资源R1不变。
C另在clean/0dB r0做rx_shuffled（同cohort固定下一个真donor）、
rx_zero、rx_oracle（原生BART源memory特权诊断）。这三者严格保持
自身图像TX及图像/自身文本噪声不变，只替换RX侧信息。
oracle与正常memory有域差异，不称理论上界，不用oracle结果给系统背书。
无匹配门控，不能以“错配会关门”制造正确优于错误的假象。

主问题：dev AWGN0，C较B的LPIPS下降；同时报告B较A和C较A。
重复先图内均值，10000次image-level paired bootstrap seed42；
单训练seed/点态95%区间，多重比较未校正。
数值筛查要求C较B、C较A及C正确较实际donor的LPIPS增益区间下界>0，
且C较B的PSNR增益区间下界>=−0.2dB。donor比较均用r0，不能和3次matched均值混比。
PSNR、SSIM、LPIPS三者完整展示，不能只报有利指标。
这个筛查不是事实成功；人工关键事实保留与新增错误审核始终PENDING，
未审核时full_semantic_success=false。

真实接收文字仅C组每档r0生成一次（相同冻结文本路径，避免三组重复生成）。
明确标注没有受到图像改善，不能当双向互促成果。
保存固定前4个dev/qual_dev样本各档/控制的真实PNG及文本，不按效果挑图。
LPIPS所有神经输出本地未重算；服务器封存FP32指标、少量PNG、checkpoint及独立
轻量/实际cache权重哈希验收，收集包不包含大数组/权重。

若仅发送端辅助有效，只能称文字引导编码；只有实际RX内容特异增益成立才称接收互助。
后续再考虑图像→文本和C/I/T。若最终仅图像业务，另需把全部资源给图像的机会成本对照；
当前不称总资源最优、连续胜数字或双向互促成功。

## 来源

- [TACO, ICML2024](https://proceedings.mlr.press/v235/lee24n.html)：文字参与编码；
  其ELIC数字熵编码/大规模训练/联合语义损失并未在本版完整照搬。
- [DeepJSCC-WZ, ICMLCN2024](https://arxiv.org/abs/2310.04311)：多尺度侧信息解码；
  论文为相关立体图像，不是本项目同址图文编码的Wyner–Ziv定理保证。
- [M3AE](https://arxiv.org/abs/2205.14204)：联合重建训练思路；
  本轮没有额外人工遮挡任务，避免又同时增加变量。
- LPIPS定义/权重：[官方固定提交](https://github.com/richzhang/PerceptualSimilarity/tree/082bb24f84c091ea94de2867d34c4544f68e0963)。
  本版独立Torch-only公式，不安装该库；权重单独离线包，不进公共仓库。
