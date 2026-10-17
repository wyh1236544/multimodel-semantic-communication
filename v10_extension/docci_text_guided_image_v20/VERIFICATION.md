# 验证状态

本地无Torch/Transformers/GPU。CPU合同、完整轻量schema负向测试、Python3.8
语法、source USTAR、独立解压与依赖冻结检查在发布回执记录实际执行结果。
Torch模型梯度/空间codec等价、真实LPIPS公式、保存权重、GPU峰值和训练效果
**本地尚未执行**；服务器check及真实GPU smoke是扩大训练的强制前提。
不能把合成TEST_ONLY fixture当模型效果。

负向测试覆盖：缺行/重复、错误source/donor、RX控制篡改TX、虚假SNR/PSNR、
被污染split、旧输出/路径越界、改变报告不覆盖。实际训练组真实梯度/参数变化、
冻结模型未变、固定训练顺序/预算、磁盘checkpoint回读都要有独立回执。

LPIPS Torch-only实现采用官方AlexNet ImageNet trunk与v0.1线性校准：
alexnet.pth SHA256
7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02；
linear.pth SHA256
df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0。
官方HTTPS文件已下载并完整哈希核对，未在本机反序列化。
服务器测试检查恒等零距离、输入梯度/冻结参数、独立functional-conv公式一致；
这不等于已经与安装的上游lpips包做数值对照。
权重原始许可/来源见官方
[PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) 与
[torchvision](https://github.com/pytorch/vision)；不把它们改称自有训练权重。

服务端collector重新核对真实cache/末次权重字节、逐例矩阵、导频/噪声种子与数值、
训练loss和样本顺序、PSNR/MSE、文本编辑指标并重算配对聚合。
轻量包排除大数组/权重，本地接收后只能轻量复核；
所有LPIPS/SSIM神经输出没有在CPU收集时重新前向，PNG为量化展示，不冒充FP32结果。
