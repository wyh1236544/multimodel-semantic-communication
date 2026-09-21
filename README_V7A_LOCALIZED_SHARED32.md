# V7-A：Localized Narratives 共享语义增强基线

## 改了什么

V7-A 不覆盖 V6.1。它在相同的 4608 次实信道使用下，把发送 token 从
`C16/I232/T40` 改成 `C32/I216/T40`。这里的文本输入仍截断到 64 个 CLIP
token；40 是发送到信道的文本私有语义 token，两者不是同一个概念。

共享分支新增三类约束：图文公共视图余弦对齐、发送前/接收后公共表示一致性、
以及批内对称 InfoNCE。原来的单对余弦损失可能所有样本收缩到相似向量，InfoNCE
用其他样本作为负例，能够显著降低这种退化风险。

训练采用三阶段课程：

1. `shared_warmup`：只让公共流承担恢复，并加强共享语义损失；
2. `disentangle`：恢复完整传输，同时加强公共/私有正交；
3. `joint_finetune`：用最终权重联合微调。

4 epoch 筛选自动按 1/1/2 分配，8 epoch 确认自动按 2/2/4 分配。第一阶段的
`common_only` 仍是等长置零消融，不能宣称是低码率传输；它只用于优化课程。

## 上传

把补丁压缩包上传到：

```text
/data1/zw/WYH/multimodel/
```

在该目录解压覆盖代码前，建议先备份本次涉及的文件。补丁不包含模型权重或数据集。

## 运行

```bash
cd /data1/zw/WYH/multimodel
chmod +x run_v7a_localized_shared32.sh
bash run_v7a_localized_shared32.sh 2>&1 | tee logs/v7a_shared32_seed42.log
```

先检查启动输出必须是：

```text
channel_tokens=288, symbols_per_token=16, channel_uses=4608
```

测试 JSON 除 PSNR、SSIM、token accuracy、BLEU-2 外，还会记录公共图文检索
Top-1、接收端公共语义到图像/文本视图的 Top-1，以及公共嵌入标准差。

## 筛选判据

与 `ln_v61_baseline_seed42` 在相同 500 个验证样本上比较。主要看：

- `full - no_common` 是否明显大于旧模型；
- `common_i2t_top1` 与 `common_t2i_top1` 是否高于随机水平 `1/N`；
- `receiver_common_*_top1` 是否随 SNR 上升而改善；
- `common_embedding_std` 是否没有趋近于零；
- full 模式 PSNR 与文本指标是否没有因减少 16 个图像私有 token 而大幅退化。

只有 V7-A 通过这些筛选后，再实现 `C32/I224/T64/S16` 的扩带宽 synergy
版本，以及真正按模式缩短 payload 的 shared-only/image/text/full 传输。
