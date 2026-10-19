# V22：先证明辅助可用，不再直接扩大融合训练

冻结日期：2026-10-19；状态：代码/协议，真实模型结果尚未运行。
该协议在看到V20/V21负结果后制定，V22为探索性验证，不称事前注册或独立最终测试。

## 问题和停止条件

1. 图像→文本：联合预训练模型是否优于原接收文本、图像独立描述、同一模型的空白图像语言改写；正确图像是否优于真实错配？caption-only胜过差文本本身不是互促。
2. 文本→图像：同一个冻结img2img模型、同样随机数/步数/强度，正确接收文本能否优于无提示、真实错配及原接收图像？美化本身不是事实恢复。
3. 图像连续/数字：同4128复符号、总单位能量AWGN下，分别看JPEG/BPG+CRC+QC-LDPC+QPSK/16QAM，不假定连续必胜。

没有 `train` 命令；没有自动训练、FSO、重试、换批量、升级依赖、下载权重或终止他人任务。
任何工程/资源/输入错误停止并保留partial。单次阶段明确预算，拥有的子进程组超时后终止。
完成只是工程完成。事实审核PENDING时，`semantic_success=false`始终保留。

## 输入、划分、模型

只读复用完成的V20 Fix1 A_self最终权重及原cache，SHA固定在V21来源代码。
真实旧0dB/r0的两张锚点须TX/观测SHA精确、指标1e-6重现，才允许新增噪声推理。
VAE/BART/V18/图像codec冻结，保存图文实际接收结果，不用原文修复接收。
dev来自原V20的100张官方TRAIN留出图片；已被反复使用，不称新test。
固定哈希排序，smoke2/quick8/pilot32，独立噪声次数1/1/3。没有训练种子搜索。
clean、AWGN0dB、AWGN−5dB；−5是当前0–20dB训练分布外推理，非负SNR带噪训练结果。
重复是独立评价试验，每次只发送一次，不是三次发送后合并恢复。

源图/源文本只进入TX、损失历史和单独评价文件；辅助只能取同次接收图像/接收文本。
512×512原letterbox保持不变，valid rectangle仅用于评价，不能作为免费接收端源mask。
每次记录每模态名义/导频估计/实现SNR、观测哈希，不将图像PSNR叫信道SNR。
两个模态同档AWGN、独立加性噪声，尚未模拟FSO共同衰落或实际光学波形。

## 图像数字对照

每图4096复符号，包括数字头与正文；另32导频，合计4128。
若再使用收到的文字辅助图像，另计9216文字载荷+32文字导频，合计13376。
正文QPSK/Gray16QAM×QC-LDPC N648 R1/2或3/4，四组全报，不按本次正确答案选最优MCS。
导频/648符号双副本QPSK R1/2头、包头CRC、正文CRC、FEC填充与循环rate matching全部计入。
16QAM正文逐帧单位能量增益以float32写入受CRC/FEC保护的头，RX不免费读取TX增益。
全帧平均能量误差<=2e-6，不归一化接收噪声；与连续图像共用同一组实际AWGN样本。
源编码按固定512/256/128/64/32分辨率优先、再JPEG95→1或BPGQP0→51，选择首个完整能装入帧的码流。
仅依据字节长度选源编码，不按SNR/PSNR挑图；明确报告尺寸/质量/字节数/是否降采样，不冒称完整高清恢复。
低码率JPEG是便于现环境直接执行的参考，不能代替强BPG数字基线，也不声称当前MCS/译码器最优。
BPG通过显式、项目内、固定SHA的Linux bpgenc/bpgdec执行；缺失拒绝，不安装、不偷偷替换成JPEG。
CRC失败固定输出RGB128，失败仍纳入指标；不丢弃难样本。未知post-FEC BER记null，分母0，不伪装零误码。
本基线采用整图CRC擦除，不是渐进/分层/分包数字最优方案；低SNR交叉不能外推所有数字方案的极限。
可将同一enhance程序应用于一个显式数字MCS，复用相同文字、生成参数和随机种子。
只有双方获得同样后处理和辅助预算，才能比较整个生成系统；纯数字与生成连续不能直接归因为通信增益。
本版不设置64GSa/s/RRC/FSO物理速率，没有吞吐/带宽/真实光链路优势结论。

## 冻结预训练推理

caption适配器为现有Transformers4.30支持的BLIP-2，显式本地目录和封存文件哈希；不给模型执行远端自定义代码。
五组：原接收文本、图像独立描述、正确图像+接收文本、错配图像+同一接收文本、空白图像+同一接收文本。
贪心、最多512新token；输入全文不截断，超上下文明确失败。输出达上限记录，不误称自然结束。
BLIP-2处理器可缩放/裁剪，512输入不意味着内部保留全部高清细节。模型可能幻觉，不保证完整恢复长描述。

enhance适配器为原版SD1/2 StableDiffusionImg2ImgPipeline，不是ControlNet或DiffJSCC论文复现。
固定30步、strength=.25、guidance=5、每条件一个候选；同一实际噪声seed对照，不看原图挑种子/强度/最好候选。
30为scheduler请求步数，img2img会按strength缩短实际去噪；另记录实际callback步数，不冒称30次UNet调用。
正确和错配提示均为V18实际接收文本，非原文oracle，空提示为独立对照；保留原图像接收基线。
CLIP77显式按75内容token分块、各自编码、拼接全部块，最多16块，超限报错。
这是未训练的长提示启发式，不是Long-CLIP或保证完整语义理解。所有组记录完整提示哈希、块数、token数。
不关闭本地pipeline自带安全检查；黑图/拒绝保留并报告，不重采样。保存512原canvas，不秘密修边/拉伸。

## 评价先定、失败不改指标

图像：同一保存uint8 PNG、valid rectangle的MSE/PSNR与固定独立LPIPS-Alex v0.1。
LPIPS必须通过单独score实际运行，未测即NOT_RUN；不以像素结果冒充感知或事实收益。
相同图片的多噪声重复先篇内平均，10000图片级bootstrap，单训练种子。
−5dB预先指定三个主LPIPS对照：正确提示对原接收、无提示、错配提示，Bonferroni alpha=.05/3。
候选数值标准：三者调整区间下界>0，正确对无提示LPIPS至少相对改善2%，PSNR下界>=−.5dB。
clean/0dB其他比较为探索性点式区间，不夸大多重比较显著性；数值通过仍不替代事实审核。
文本：旧normalized WER保留辅助；按物体/属性/数量/关系逐事实判断保留率及不支持新增事实。
BERTScore/CAPTURE/AlignScore本版不下载、不执行，标NOT_RUN；不要用BLIP-2/CLIP自评替代独立审核。
先10图×3档、只r0的人工表，source事实先写，再盲评输出，保留PRIVATE解盲表。
图像关键事实需要盲看原/恢复图逐项审核；不能因LPIPS更好就判定“红色/数量/身份”更准确。
预训练数据重叠须记录具体checkpoint来源/commit和训练集说明，unknown就是unknown。
DOCCI微调的captioner不能当干净独立证明；旧发布时间也不构成零重叠证据。

## 一手依据和兼容边界

- [DiffJSCC](https://arxiv.org/html/2404.17736v3)：接收图像生成文本条件，另训练控制模块；不是零训练直接接SD或独立图文互促成功的先例。
- [BLIP-2 ICML2023](https://proceedings.mlr.press/v202/li23q.html)：复用预训练图文连接能力。
- [Transformers4.30 BLIP-2 API](https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/blip-2)。
- [Diffusers0.16 img2img API](https://huggingface.co/docs/diffusers/v0.16.0/en/api/pipelines/stable_diffusion/img2img)：本地版本仍需实测，不升级来凑兼容。
- [BPG作者实现](https://bellard.org/bpg/)；LDPC数字表及来源沿用V19 PROTOCOL，不冒称完整WLAN PHY。

本机没有Torch/Transformers/Diffusers，CPU合同/合成集成只能验证代码边界，不能证明真实GPU或生成效果。
