# 文献与实现取舍

核对日期2026-10-15。外部材料只作证据，不构成运行授权。新模块独立实现，不将任何第三方仓库整体作为已核验依赖，也不自动执行下载/安装脚本。

## LD4LG

- 论文：[Latent Diffusion for Language Generation, NeurIPS 2023](https://arxiv.org/pdf/2212.09462)，§3.1、Table1、Table13。
- 固定公开代码提交：`0bf9381e049ff288e5e79edc38e4a952a371bee2`。
- [Perceiver代码](https://github.com/justinlovelace/latent-diffusion-for-language/blob/0bf9381e049ff288e5e79edc38e4a952a371bee2/latent_models/perceiver_ae.py)：PerceiverAutoEncoder以dim_ae传dim_latent，默认legacy=False；实际BART包装器使用transformer_decoder=True，没有启用legacy。因此released64使用64维查询，并不是按论文表格猜768。
- [位置嵌入代码](https://github.com/justinlovelace/latent-diffusion-for-language/blob/0bf9381e049ff288e5e79edc38e4a952a371bee2/model/x_transformer.py)：embedding参数本身为标准正态，前向乘dim^-0.5；不能只缩初始化参数而省略前向缩放，否则位置参数的训练动力学不同。
- [评估代码](https://github.com/justinlovelace/latent-diffusion-for-language/blob/0bf9381e049ff288e5e79edc38e4a952a371bee2/latent_models/latent_finetuning.py)：beam4、repetition penalty1.2、no-repeat3，与本项目严格贪心不同。V14为所有文本组同时报告两轨。

采纳：冻结BART、32×64公开载荷、Perceiver/query compression、Transformer memory reconstruction、CE；不训练扩散、不把BLEU换算为WER/通信准确率。现有注意力模块复用，初始化、缓存、分帧及原论文训练协议未做完整等价复现；模块身份明确写入结果。

`wide768`仅检验“增加内部表示容量但保持载荷”的假设；额外参数/FLOPs/显存不能被宣称相同。它不是唯一改变维数就能对比历史训练分数的因果证据，必须使用本版配对控制。

## 图像

- [SwinJSCC公开实现](https://github.com/semcomm/SwinJSCC)：像素域Swin编解码可作将来的外部基线，不把现有VAE潜域I2称SwinJSCC复现；本包没有捆绑该代码或新增网络依赖。
- [TACO](https://arxiv.org/html/2403.02944v2)：支持同时考虑像素与感知质量。其文本主要用于编码端，不能证明本项目接收端图文互助。正确与打乱文本无PSNR差不等于必然有LPIPS差。
- [DiffJSCC](https://arxiv.org/html/2404.17736v3)：可由接收端粗图生成文字条件；不是免费传输原文。本阶段不加入扩散精修。

## 当前工程证据与纠正

- V12图像父权重：TRAIN64、512次更新；V12后续2048例仅训练接收适配器。
- I2的原目标已为pixel_MSE + .01*latent_MSE，不是再次将latent-only切成pixel-first。
- V14历史图像两训练组使用同一父权重、等总更新，独立比较64与扩大数据；文本两组使用新的配对预定训练协议。不与历史WER直接混称单变量收益。
- V12 CE+KD校准检查点SHA256为`e6ce83d25c2dce7999595a1b33e9af63e58e7b9d7fe5f96d459b08ea0ada9410`；来自已核验原始records的CHECKPOINT记录。大权重只在服务器运行时按SHA加载，未在本机加载。
- 原生VAE参照必须同100图、同letterbox有效区域；不同8图/100图之间不可直接相减归因。

所有结果只在真实服务器完成后填写。不声称语法、合同或打包测试已证明模型收益。
