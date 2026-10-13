# V6依据与非复现边界

V5正式附件核验另存于本地clip_bart_joint_v5_review分析目录；
服务器原V5_02保持不变。3000text CE仍下降，
32validation matchedWER.465→joint.447；Common对齐损失.773→.279，
J1-J0 PSNR+0.00584dB/J1-J2文字224/224相同/关闭融合cleanPSNR+0.125。
这些是本项目结果，不是论文收益或全领域结论。

- ALBEF: Align before Fuse (NeurIPS2021), https://arxiv.org/abs/2107.07651 .
  借鉴对齐再融合；V6不复制momentum teacher，不宣称复现VQA/retrieval成绩。
- DiffJSCC v3, https://arxiv.org/html/2404.17736v3 .
  粗JSCC图片+接收端提取文字/空间信息+CSI条件，冻结SD并训练控制模块。
  V6生成只是小型提示残差的TRAIN嵌入回归，绝不是其控制模块/扩散训练。
  其主要AWGN/Rayleigh、CSI设定和图像尺寸/码率不等于本项目FSO。
- SGD-JSCC v1, https://arxiv.org/html/2501.01138v1 .
  IV-A假设文字完美传输并忽略文字开销；不可用于本项目免费原文提示。
  其DiT从大规模数据训练的设计不是直接可替换的本项目解码器。
- V4 Perceiver-inspired query codec retained: ../bart_latent_codec_v4/SOURCES.md .
  No BART/CLIP/VAE backbone unfreeze, no additional full model downloads.

Common-only caption supervision, equal16/16 pooling, bounded coarse/caption
residuals and private-feature dropout are project hypotheses. TRAIN dropout is
not a measured FSO model. A small semantic adapter can still hallucinate; all
generated candidates and failures remain visible. No new novelty/publication claim.
