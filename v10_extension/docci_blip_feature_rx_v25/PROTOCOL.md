# V25 预注册：接收图像的 BLIP-2 特征辅助文本恢复

状态：候选实现，无新真实训练结果。2026-10-19。只实现图像→文字，文本→图像和双向循环不在本轮启动。

## 假设与结构

正确配对的接收图像，在文本受损时含有可用补充信息；冻结BLIP2的32×768 Q-Former特征可经小适配器改善冻结BART的恢复。
这是可证伪假设，不是BLIP2→BART已预训练对齐。Q-Former原来对接OPT，不保证对BART有效。
不把V24两张clean主体识别成功当作联合恢复成功；灰图幻觉和低SNR主体误认均保留。

图像走已完成V20 A单路→VAE→实际uint8接收图→固定BLIP2视觉编码器/Q-Former→缓存32视觉token。
没有原图输入BLIP2，没有生成描述/原文/真实长度输入适配器。32视觉token不是新传输载荷。
文本走冻结V18 noise_replay→AWGN→原RX→9×32×768接收memory。
原BART接收memory不替换；末两层原cross-attn的norm之后、FFN之前，追加256宽门控视觉cross-attn。
门标量初始化0；注意力权重非零，后续组梯度必须真实非零且参数更新。无匹配门，不根据错图决定退回基线。
内容软门仅来自原文本接收器对[decoder_start,BOS]之后的EOS预测，未使用源段数、mask或gold EOS。

两组适配器均1,516,290参数，同初始化、同更新数、同批次和噪声观测：

- self：只读取自身接收memory和软门，固定加权成32槽，完全不读取图像。
- visual：读取实际接收图片的冻结Q-Former32槽。

仅新适配器学习；图像codec、文本codec、VAE、BART、BLIP2及OPT全部冻结。优化CE，不新增随机语义教师/匹配损失。
不让“修复后的文字”回灌图像，本轮也不新增Common或文本引导扩散。

## 数据、信道与预算

复用已固定V20全缓存：8192 TRAIN与100 dev。固定种子排序选择smoke8/2、pilot512/32。
TRAIN/dev图像字节和文本哈希不交叉；dev是反复探索过的官方TRAIN来源留出集，不是独立最终测试。
不读取官方test/qual_test，不按输出筛样本。BLIP2预训练污染状态unknown，不宣称无重合。

每图一次传输R1，图像4096+32导频=4128复符号，文本9216+32=9248，总13376；无新增发送或免费原文。
base/self没有图像内容输入，但对照仍保留相同整帧预算，表示“已分配但未使用”，不能把额外算力称为免费。
native caption-only只使用4128图像帧，其自身预算单列；不能说与联合有同样的信息输入。

缓存每图image/text各clean、0、−5的独立噪声，沿用V22明确的单位增益AWGN和32复杂导频。
单位功率、无噪后归一化。不是FSO、不是实际波形/64GSa/s速率实验。clean无新增噪声但有codec失真。
五种评价条件：clean=(clean,clean)，both0=(0,0)，text0=(clean,0)，text_m5=(clean,−5)，both_m5=(−5,−5)。
smoke只跑前三种。训练仅clean/both0/text0；−5是冻结codec及新适配器训练范围外压力测试，不作为事后挑选的主要条件。
text0是异质可靠性机制诊断，不能冒充同一时刻共用同一FSO衰落的真实光链路。
本轮是小适配器对带噪接收观测的学习，三个条件的噪声缓存跨epoch复用；不是新鲜噪声逐批采样的端到端JSCC训练，不更新源/信道编解码器。

## 训练与评价矩阵

smoke：每组3更新，8条TRAIN最多实际使用6条，只验证真实权重/梯度/保存/生成。smoke权重不续用。
pilot：每组512更新，512条TRAIN，4轮，微批1/累积4，末次固定checkpoint。两组共1024更新。
固定LR1e-4带预定warmup/线性尾部，AdamW0.01，梯度裁剪1，无AMP或自动batch变化，不按dev选优。

每个dev/条件输出8行：原文本base、BLIP2 native caption-only、self，以及visual正确/真实不同样本错配/灰图特征/零特征/旁路关闭。
错配只替换视觉token，不动自己的文本memory；donor固定同cohort轮换。灰图由同BLIP2真实编码，不等于零token。
旁路关闭必须逐例与base生成token完全一致。相同图像条件的caption-only复用，不当作独立生成证据。
smoke48行，pilot1280行。全部保留，生成仍greedy，未新增禁止重复、长度提示或强制EOS。

主比较预先固定both0。词面numeric screen同时要求：

1. visual正确比self平均normalized WER至少低2个百分点，图片配对95%区间下限>0；
2. visual正确比错图低至少1个百分点，配对区间下限>0；
3. visual正确优于原冻结文本base，配对区间下限>0；
4. 相邻重复和源空段误激活不多于self。源空段只在评价后计算，不用于接收。

10000次图片级配对bootstrap，n<3不计算区间。单训练seed、复用dev、点式区间未多重比较校正，不称最终显著结论。
WER不等于语义准确率，caption-only短句不能与长原文仅凭WER裁决。另输出独立人工事实审核模板：物体、属性、数量、关系、无依据新增。
只有联合同时在事实保留上胜过文本单路和看图描述单路，才讨论互促。自动报告始终facts=PENDING、semantic_success=false。

## 工程与停止条件

原Torch1.12.0/HF4.30.2，物理GPU1，启动空闲20GiB、运行余量4GiB、项目磁盘启动16GiB/运行4GiB。
本地无Torch/HF，纯CPU协议/合成I/O检查不是真实模型验证。服务器先跑V24的6项真实tinyHF与V25的8项真实模型合同，再加载完整权重。
完整BLIP2文件按已有官方封存SHA检查；保存原始dtype说明时只转换元数据副本，不修改模型。
缓存前后冻结tensor哈希、已完成V20父权重两锚点复现、新组梯度/更新与实际磁盘roundtrip必须通过。
任一门禁失败即停止，保留现场、不重试、不安装依赖或停止其他用户任务。smoke后不自动pilot，pilot后不自动扩量/图像恢复。
max-hours为保护上限而非ETA。cache、train、evaluation、verify、collect分别计时。只有LAUNCH_FINISHED.json=VERIFIED且records存在是完整启动流程结束。

## 来源

- [BLIP-2](https://proceedings.mlr.press/v202/li23q.html)：使用预训练Q-Former视觉语义表示，不声称移植其语言模型对齐到BART。
- [固定HF4.30.2 BLIP-2源码](https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/models/blip_2/modeling_blip_2.py)：extract复制native generate的vision→query路径，在language_projection之前取特征；真实小模型合同检验一致。
- [Flamingo](https://arxiv.org/abs/2204.14198)：借鉴冻结语言模型的门控视觉注意力，不复现其模型/数据规模。
- 项目V16/V17/V20/V21负结果保留。本次不把更多训练、开放门、预训练视觉表示中的任何一项当作已经有效的证据。

PRIVATE：缓存、原始DOCCI图文、records、权重不公开；GitHub仅源码/协议/聚合状态。
