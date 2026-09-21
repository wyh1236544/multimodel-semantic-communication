# V7-B：回归优先的共享语义版本

## 设计

V7-B 保持 288 个信道 token 和 4608 次实信道使用不变，将 V7-A 的
`C32/I216/T40` 改成 `C32/I224/T32`。共享占 11.11%，图像私有占
77.78%，文本私有占 11.11%。

新增的 `common_coarse_head` 只接收经过信道后的 32 个 `Z_C`，直接预测
5x5 平滑后的 VAE latent。它在联合 Transformer 解码之前分支，因此不能从
图像或文本私有 token 偷取信息。该预测目前作为辅助回归监督，不直接叠加到
主重建结果；这是为了安全继承 V7-A，先验证共享语义能否获得粗结构回归能力。

接收端文本到图像路径使用基于 SNR 的可靠性：0/10/20 dB 时约为
0.047/0.731/0.993。图像专用解码会先压低不可靠文本信道 token，后续
cross-attention 和图像 memory 再应用同一可靠性，文本自身的重建路径不受影响。

## 上传与 smoke test

将补丁上传到 `/data1/zw/WYH/multimodel/` 后覆盖解压。先执行：

```bash
TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python train.py \
  --config configs/v7b_regression_balanced.yaml \
  --preset original_4608 \
  --output-dir experiment_results/ln_v7b_regression_balanced_smoke \
  --init-checkpoint experiment_results/ln_v7a_shared32_seed42/seed42.pth \
  --allow-shape-expansion --seeds 42 --smoke-test \
  --data-root /data1/zw/WYH/multimodel/COCO2017 \
  --vae-path /data1/zw/WYH/multimodel/pretrained/sd-vae-ft-mse \
  --text-model-path /data1/zw/WYH/multimodel/pretrained/clip-vit-base-patch32
```

确认输出为 `C32/I224/T32`、`channel_uses=4608`，且新增 coarse head 只作为
missing keys 初始化，旧参数应完整加载。然后运行：

```bash
chmod +x run_v7b_regression_balanced.sh
bash run_v7b_regression_balanced.sh 2>&1 | tee logs/v7b_regression_balanced_seed42.log
```

脚本额外生成 `eval_no_text_snr_gate.json`，可与 `eval.json` 的 full 模式做
完全相同权重下的门控消融。主要判据是 0 dB 下 full 与 no-text-private 的 PSNR
差距是否缩小，以及 `common_coarse_latent_mse` 是否随 SNR 改善。
