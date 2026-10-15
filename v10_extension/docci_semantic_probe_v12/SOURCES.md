# V12 文献依据与采纳边界

1. Qin et al., *Perceptual Image Compression with Cooperative Cross-Modal Side Information*, arXiv:2311.13847v2.
   https://arxiv.org/html/2311.13847v2 。正式期刊题名为 *Perceptual image compression with textual side information*,
   Pattern Recognition169 (2026),111848, DOI10.1016/j.patcog.2025.111848；完整方法依据来自公开作者预印本，未把所有预印本表当作最终期刊表重验。
   采纳：冻结CLIP文本编码器、图像解码中空间语义调制、重训无文本与错配干预分别评价。
   V12是独立的SSA启发模块，不复制其代码、HiFiC熵模型、SSA-GAN或判别器；没有复现其400轮COCO训练和感知率失真曲线。
   该论文在接收端有文本辅助信息；不证明本项目的连续FSO、文本精确恢复、双向互促或包含文字开销后的总资源优势。
2. Lewis et al., *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*, ACL2020.
   https://aclanthology.org/2020.acl-main.703/ 。使用项目现有冻结BART，不把去噪预训练直接等同于连续信道鲁棒性。
3. Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network*, arXiv:1503.02531.
   https://arxiv.org/abs/1503.02531 。采纳温度缩放完整输出分布KL、教师无梯度、保留真实标签CE。
   V12的温度2/权重0.5是提前固定的首轮超参数，不是论文保证的DOCCI最优值。原生memory仅TRAIN教师可见，不能发送或提供给实际RX。
4. Zhang et al., *Long-CLIP: Unlocking the Long-Text Capability of CLIP*, ECCV2024.
   https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06793.pdf 。重用已在V8发布的作者实现与官方SHA锁定权重。
   连续246正文窗口池化是本项目明确的长描述处理策略，不声称原生上下文超过248，也不保证原文可逆恢复。

本轮评价以PSNR为预先指定主指标、SSIM为辅助，LPIPS未测。因此不能借论文的LPIPS/FID收益给本项目作感知性能背书。
未宣称新颖性“首次”；检索有效、重建改善、同资源通信收益是三种不同证据。
