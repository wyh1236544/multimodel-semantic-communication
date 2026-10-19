# V23：先补图像数字曲线，再检验预训练图像辅助文本

协议日期2026-10-19，看到V22负结果后制定，探索性而非独立最终测试。
历史源码/manifest/数据/缓存/权重/结果不改。无train命令、不下载或安装服务器依赖，
不继续V20三组训练、不自动扩大、不终止其他用户进程、不改旧QUALITY_STOP。
预训练权重仅由用户在本机显式下载后上传；源码包不含第三方权重或DOCCI样本。

## A 图像连续与实际JPEG数字链路

固定V20 Fix1 A_self最终权重、V18文本codec、冻结VAE/BART，读旧cache不重建。
新prepare必须先重现两条真实旧0dB锚点：发送/观测SHA精确，旧指标1e-6。
也可CPU导入V22实际完成的两图smoke，PNG/文本逐字节保留，不重复GPU推理。
导入噪声沿用V22家族；新prepare用独立V23家族；同图连续和数字共用逐个实际噪声样本。

profile固定：smoke2图×1draw×clean/0/−5；quick8图×1draw；pilot32图×3draw。
quick/pilot网格clean/−8/−5/−2/0/2/5/10/20dB。选择顺序与V22固定dev排序一致。
dev是原V20官方TRAIN留出100的固定前缀，已经复用，不称独立test。
负SNR是0–20训练范围外的冻结推理，不冒称新增负SNR训练或FSO。
每次R=1，draw是独立评价实现，不发送三次后合并；保留失败。

图像4128复符号=4096载荷（数字头/正文计入）+32导频，全帧平均功率1。
复用V22实际JPEG字节、CRC、N648 LDPC1/2或3/4、Gray QPSK或16QAM四MCS，
含头双副本、填充、rate matching及正文增益保护。不根据本批原图指标选最优MCS。
源编码只按字节长度选择最高可装分辨率/质量；512/256/128/64/32、JPEG95→1。
降采样、质量、包字节量、preFEC BER、条件postFEC BER、CRC失败均报告。
整图CRC失败固定灰图，并纳入PSNR/LPIPS；不是渐进分包/最优数字接收。
JPEG只是现环境低码率参考，不代表最强数字，更不等于最强BPG。
本版未新增JPEG2000或BPG执行入口，不把它们写成已执行；旧V22 BPG仍保留。
数字源编码/收发/独立重放核验用CPU，LPIPS仍由单独GPU score实际运行。

## B 预训练图像→文本：六组，不训练桥

只在clean/0/−5三条件生成文本，其他图像曲线点不额外生成BART文本。
六组固定：原V18接收文本；仅接收图像caption；接收图像+接收文本；
真实错配图像+同一接收文本；灰图+同一接收文本；同一接收图像+真实错配接收文本。
增加wrong_text是为检验两路内容依赖，不能只靠提示任务差异或caption替代文本。
错配使用同条件/同draw其他样本；不存在原图、原文、长度或源mask进入实际receiver。

完整系统13376复符号=图像4128+文本9248；图像单独caption参考只需4128，
另列其资源，不把“只传图片生成描述”当作原始文本传输或图文联合传输。
没有自动符号/功率重新分配，没有实光链路波形/带宽/时间优势主张。

模型为官方Salesforce/blip2-opt-2.7b固定2023-05-24快照
1ef7f63a8f0a144c13fdca8103eb7b4691c74cec，避开2024 image-token配置迁移。
11文件共约15.5GB；官方Git blob/LFS SHA逐文件验证。服务器加载half，
不需要缺失的SentencePiece，也不升级Transformers4.30.2来匹配当前main。
来源版本确定不等于DOCCI零重叠，overlap_audit保持unknown。
OPT版本不是保证遵循指令的恢复专家；失败可能是captioner/提示/任务边界，
不直接证明所有图文互促不可行。processor可能缩放/裁剪，不称原生512细节保留。
贪心、不采样、不禁重复、最多512新token；完整输入越界停止，不截断。
EOS/capacity stop记录，不偷偷延长或换种子。冻结参数运行前后校验。

主问题−5dB：joint_correct关键事实F1是否同时超过received_text、caption_only、
joint_wrong_image、joint_blank_image、joint_wrong_text，且新增错误事实不增加。
主事实类型：物体、属性/颜色、数量、位置/关系。先从原文建立事实清单，
再盲评输出；事实未完成人工审核始终PENDING/semantic_success=false。
WER保留辅助；BERTScore/CAPTURE/检测/VQA评分本版不伪装执行、不自评。
若32图才进行数值screen，建议五个事实主对照同时区间及2pp候选最小增益，
须在独立事实清单冻结后另行登记；本版不会仅用WER自动通过。
两图smoke只检查实际流程和明显失效；不以n=2 bootstrap宣布显著。

## C 潜变量校准，不运行扩散

只读V20 TRAIN，固定哈希排序8或32张，每图clean/0/−5一次。
先A_self真实锚点，通过后比较z_rx−z_source与z_rx−z_clean，报告均值、MSE、
标准差、偏度、超额峰度、空间相邻相关、逐通道MSE，均为描述性统计。
原latent用于receiver执行之后的诊断，不传入实际接收恢复。
另比较同一个已冻结sd-vae-ft-mse的decode→uint8量化→encode均值→decode，
量化包含在往返损失内；不是对独立SD1.5 VAE权重身份的一致性验证。
保存source/cleanRX/noisyRX/reencoded四个latent在服务器NPZ中，collector不带走。
这些误差不是已经证明的独立高斯噪声；timestep_mapping始终null。
没有SNR直接映射扩散步数、UNet去噪、回拉参数扫点或新增训练。

## 核验与预算

每阶段新目录，run和独立CPU verify共享launch显式硬预算，超时只停自有PGID。
缺资产/显存/矩阵/哈希/包回环失败停止，保留partial和日志，不自动改批量重试。
GPU1起始至少20GiB空闲、运行保留4GiB；这是保护阈值，不是实测峰值。
quick/pilot必须先提供相同阶段实际smoke；caption还要同一个asset lock。
PSNR/MSE从同一uint8 PNG有效区域独立CPU重算；LPIPS-Alex v0.1需GPU实测。
同图重复先平均再跨图；固定单训练seed、复用dev，不能覆盖训练不确定性。
所有原始图文/records保持PRIVATE，GitHub只保存源码、协议、聚合及交付状态。

## 一手依据与适用界限

- BLIP-2旧版API：https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/blip-2
- 固定官方模型：https://huggingface.co/Salesforce/blip2-opt-2.7b/tree/1ef7f63a8f0a144c13fdca8103eb7b4691c74cec
- VAE兼容说明：https://huggingface.co/stabilityai/sd-vae-ft-mse （仅解码器微调，仍须本地尺度/权重核验）
- DPS：https://arxiv.org/abs/2209.14687 （逆问题观测约束依据，非本版已实现算法）
- DiffJSCC：https://arxiv.org/html/2404.17736v3 （另训练控制模块，非直接img2img复现）
