# V8-A：可独立验证的共享语义基础层

V8-A 不再把 `common_only` 实现为 `C32 + zero(I224) + zero(T32)` 的
等长消融。启用 `true_common_only_transmission` 后，信道中只存在 32 个
Common Token，因此实际信道使用量从完整模式的 4608 降到 512，私有位置也
不会在 AWGN 后变成噪声并污染联合解码器。

接收端新增三个只读取物理 Common 载荷的输出：

- `common_coarse_latents`：共享粗图 latent；
- `common_base_text_logits`：前 24 Token 的核心文字恢复；
- `common_receiver_semantic`：对齐冻结 CLIP 文本表示的语义向量。

## 服务器结构检查

```bash
TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python check_v8a_common_base.py \
  --data-root /data1/zw/WYH/multimodel/COCO2017 \
  --vae-path /data1/zw/WYH/multimodel/pretrained/sd-vae-ft-mse \
  --text-model-path /data1/zw/WYH/multimodel/pretrained/clip-vit-base-patch32
```

预期打印 `common payload: C32 x 16 = 512` 和
`full payload: C32 + I224 + T32 = 4608`。

## Smoke test

Smoke test 的单个 epoch 会命中最终联合阶段，主要检查新参数加载和完整前向。
结构脚本则专门检查真实 C-only 路径。正式训练执行：

```bash
chmod +x run_v8a_common_base.sh
bash run_v8a_common_base.sh 2>&1 | tee logs/v8a_common_base_seed42.log
```

`eval.json` 中重点查看：

- `common_semantic_text_r1/r5/r10`；
- `common_caption_token_accuracy` 和 `common_caption_bleu2`；
- `common_coarse_latent_mse`；
- `channel_uses`，其中 `common_only` 必须为 512。

`common_only_shuffled` 将同一批次的 Common 载荷循环错配，
`common_only_random` 使用同形状随机载荷；两者也只使用512次信道。如果正确
Common 的检索、核心文字和粗图显著优于这两个对照，才能把效果归因于传输的
共享语义，而不是接收端解码器先验。

生成的 `common_only_panels/*.png` 会直接展示 Common 粗图，`samples.csv`
同时保存 Common 核心文字。生成模型不在 V8-A 中启用；只有当检索、文字和
粗图的因果消融证明 Common 载荷有效后，再将核心文字接入扩散模型。
