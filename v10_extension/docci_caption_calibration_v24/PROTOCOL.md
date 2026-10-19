# V24：BLIP-2 两图基础调用校准

状态：新代码，不是新训练模型。协议在 V23 失败输出被检查后制定，属于探索性诊断；
不把本协议称为事前注册的独立测试。旧 V23 源码、manifest、权重、PNG、文本和结果不修改。

## 问题与边界

V23 两图的 clean caption-only 都为空；0 dB 联合部分复制受损文字，−5 dB 长循环。
这不证明所有图文互促不可行，也没有证据认定提示是唯一根因。
本次只判断基础看图调用是否可用：预处理→视觉编码器→Q-Former→语言投影→OPT→EOS。
不训练、不下载、不安装升级、不重新执行 V20/V18 传输、不继续原512-token复杂纠错。
没有新增 Common、FSO、扩散、光波形或实际速率实验；不启动服务器作业。

官方固定版本依据：

- [Transformers 4.30 BLIP-2 文档](https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/blip-2)：无文字提示的 caption 与简短问答是分开的例子。
- [实际 4.30.2 generate 源码](https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/models/blip_2/modeling_blip_2.py)：无 input_ids 时内部使用 BOS，视觉查询与提示嵌入后调用 language_model.generate。
- [BLIP-2 原论文](https://proceedings.mlr.press/v202/li23q.html)：已预训练视觉语言连接器，不是专门训练的受损长段落纠错器。

## 输入与调用矩阵

仅接入已完成 V23 smoke：2个复用 dev、每个 clean/0/−5 AWGN 各一个独立信道实现。
clean 为没有额外信道噪声，仍经历 codec；不是原图、20 dB 或真实光学 SNR。
−5 dB 为原父模型训练范围外的冻结推理。DOCCI TRAIN 来源的既有 dev，不是最终独立测试。

caption 共20行：每例1个未传输原图诊断、3条件×接收/真实错图/常量灰图。
未传输原图标记 privileged，绝不作为真实接收组输入或等资源方案。
模型请求不包含源描述、源长度、掩码、原文答案或旧接收文字。原文只在 PRIVATE 展示中供评价。
完全相同图片字节+提示+精度只计算一次；实际本包通常为9次独立调用。
同一 donor/灰图复用有显式 inference_id，不计作新的独立样本或图像不敏感证据。

caption 省略 text 参数，固定最多64新token；greedy、beam1、不采样、不禁止重复、不强制EOS。
初始 dtype=float16。float32 仅独立显式诊断，起始36GiB空闲阈值，无自动精度切换。
原图已是保存的512全视图参考；BLIP-2仍使用固定预训练 processor 尺寸，而非声称原生512视觉编码。

QA 为另一个显式操作：同20个视觉条件×4个固定通用问题=80行，通常36次实际调用。
object/color/count/relation 问题写死，不从原文生成问题或提供对象名/正确答案。
最多32新token。只有两个原图及两个clean接收图的真实caption通过人工逐项审核后可运行。
审核只授权 QA，不授权扩大数据、训练、新下载或宣称语义成功。未通过则停止此配方。
不能把所有图片的短答案当成可恢复整篇DOCCI描述的证据。

## 遥测与身份

完整返回token、按实际 score步数分离出的新token、原始未去特殊token的解码、EOS和容量停止。
不能因为 OPT 的初始BOS可能与EOS同ID就把初始token计为自然结束。
处理后张量尺寸/范围/哈希，真实vision/QFormer/projection/首步LM有限性摘要，
逐步top5概率与EOS概率。processed scores 的合法−inf单独计数，NaN/+inf失败；原始LM logits必须全有限。
模型加载缺失/额外/形状错误键均失败。模型全state前后SHA、冻结/无梯度；
原官方2023资产全文件流式哈希在反序列化前验证，结束仅stat对照，不伪称重复全文重哈希。
记录真实GPU allocated/reserved峰值、起始空闲、实际调用耗时；峰值不等于总进程显存，阈值不等于预测峰值。

保留全部空输出、提前EOS、循环和容量停止。邻接重复及重复8gram仅为诊断，不代替人工事实评估。
完整20/80行与控制输入/输出SHA、概率trace、源码快照和原prepared副本由CPU验收。
工程VERIFIED不等于 factual usable 或 semantic success；facts始终PENDING，semantic_success始终false。

## 决策规则

1. 先caption，不自动QA。检查两个原图和两个clean接收图是否非空、切题、主要物体有图像依据、无长循环。
2. 原图也失败：优先排查调用、token/EOS/预处理/数值；不继续长训练或立即下载更大模型。
3. 原图正常而接收图失败：检查codec/信道退化及细节可辨性；不把此差异归咎语言训练量。
4. 通过人工基本可用性后，才运行QA；正确图/错图/灰图差异仍须人工判断事实，不能仅按字符串不同判成功。
5. 两图仅诊断，不做显著性/泛化推断，不自动扩到8/32，也不启用新融合层。
6. 后续互促必须同时超过接收文字与仅图像两个单路，并超过错配/灰图，且不增加错误事实；本版没有实现该融合。

原信道资源仍为image4128、text9248、合计13376复符号，R1；本诊断不新增传输。
原图特权诊断不计入公平系统比较。接收端BLIP-2计算/时延必须另报，不是免费计算。
预训练数据与DOCCI重叠未知，不能声称无污染或独立测试。

## 运行保护与隐私

默认物理GPU1，float16起始20GiB空闲、运行4GiB余量；不终止他人进程、不自动调batch/重试。
30分钟是整个launcher停止预算，含CPU/HF合同、资产哈希、推理和收集，非ETA。
真实tinyHF CPU合同必须通过，再加载真实模型。缺Torch/HF不跳过服务器门禁。
本地无Torch/HF，所以本地合成通过不能冒充真实HF/GPU/新恢复表现。
PRIVATE collector在独立新路径复制并验证，USTAR成员逐字节复核，源run只读。
不调用存在未导入seed问题的旧V23 caption renderer；旧模块不修写。
公开只保存新源码/协议和必要聚合；模型权重、完整DOCCI图文、rawrecords、私人HTML不进GitHub。
