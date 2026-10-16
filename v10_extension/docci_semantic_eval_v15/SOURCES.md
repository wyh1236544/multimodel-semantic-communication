# 一手依据与采纳边界

核对：2026-10-16。以下是方法选择依据，不是本项目已实现或复现的成果。

- [BERTScore官方重缩放说明](https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md)：线性重缩放改善可读性，不提高事实区分能力。当前Longformer没有在本协议核验的官方baseline表，保留raw P/R/F1与真实对照；不能把raw分数或锚点比值叫准确率。
- [CAPTURE论文](https://arxiv.org/html/2405.19092v4)、[官方评分实现](https://github.com/foundation-multimodal-models/CAPTURE/blob/main/capture/capture_metric/capture.py)：详细caption解析物体/属性/关系；适合作候选辅助评价。merge_sentence_results没有把entity.quantity作为独立计分事实，属性评分将不同物体的属性合并，可能漏数量或颜色归属交换；语义软匹配也不保证事实正确。需要受控反例与人工审核。上游默认SentenceTransformer CUDA加载及FactualSceneGraph/NLTK等资源尚未适配本服务器；不宣称安装一个pip包即离线就绪。
- [AlignScore ACL2023](https://aclanthology.org/2023.acl-long.634/)、[官方实现](https://github.com/yuh-zha/AlignScore)：原文作为context、候选作为claim的方向性支持评分；不是校准的矛盾概率，也不衡量遗漏。基础模型、task checkpoint、tokenizer与spaCy资源需完整离线核验；官方整套benchmark依赖不可直接装进旧Conda。本版仅有受校验的导入接口。
- [ICAE ICLR2024](https://arxiv.org/abs/2307.06945)、[PLANNER NeurIPS2023](https://arxiv.org/abs/2306.02531)：支持连续压缩表示需要充分学习解码接口的动机，任务/骨干/数据/资源不同，不保证本项目大数据预训练或LoRA一定有效。
- [LD4LG](https://arxiv.org/abs/2212.09462)：V14沿用公开BART代码默认窄64约定的小控制，不是整篇论文复现；Table13内部768不可当成公开默认。详见V14 SOURCES。
- [DeepSC](https://arxiv.org/abs/2006.10685)：带噪端到端文本语义通信与退化曲线的参考；不等同DOCCI长描述或实际FSO链路。先clean保真，再单独研究AWGN/FSO，同资源数字比较仍不可省略。

Long-CLIP不作主评价器，避免用本模型语义教师给自身打分。人工审核是原文支持的内容核对，不等同图片事实真值；不能将不受原文支持却可能真实的视觉描述强行标成视觉幻觉。
