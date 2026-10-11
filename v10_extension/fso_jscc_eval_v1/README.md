# 冻结权重扩大验证 V1

这版不训练、不更新任何模型权重，只核实本次 FSO 带噪训练的收益能否推广到更多验证图片和详细光链路。
仅新增 `v10_extension/fso_jscc_eval_v1/`；共享 HANDOFF 先备份后追加。不修改旧源代码、结果、数据、划分、环境及权重。

## 实验设计

- 四组沿用已完成权重：M0 原 A；M1 clean 微调；M2 AWGN 微调；M3 FSO 微调。模型结构/宽度相同，三组已训练步数相同。
- 128 张已有验证划分的固定前缀：旧 32 张方向筛选图片和新增 96 张分别报告。新增只针对这次筛选，不保证项目全部历史从未使用。
- 单次发送 R=1。每个信道条件两次独立随机试验，用于估计稳定性，不对接收信号做重复合并。没有新训练种子。
- 可微信道代理：clean + AWGN/弱/中/强 FSO，名义 SNR −10/−5/0/5/10/20 dB。共 25,088 次。
- 详细 V3 波形：good_p09、moderate_p24/p21/p18/p15、stress_p09；全部 128 图 × 两次试验 × 四组 = 6,144 次。
- 总共 31,232 次前向验证；预检小验证仅 128 次。
- 预先规定主结果：**新增 96 图**的 M3−M2 PSNR，先对每图的 moderate_p24/p21/p18 和试次平均，再做 5,000 次图片 bootstrap。旧 32 图不参与这个主结果。
- 继承原物理参数：64GSa/s、4SPS、16GBd、RRC 滚降 .1、原光功率/湍流/指向误差/V3 同步接收配置，不借此更换链路或提高资源预算。

对照的图片、文字和光场/噪声随机数严格配对。所有失锁样本都纳入指标。
原 VAE 后验有随机采样；新流程先实测一次零输入采样的 CPU/GPU RNG 消耗是否等于同形状高斯采样，再重放旧训练缓存的 2,000 次 RNG 消耗，不读取训练图片、不训练。随后验证图片使用原随机状态。保存潜变量哈希和 RNG 证据；契约不一致则停止。
记录 PSNR、SSIM、文本 WER/CER/BLEU/token accuracy、导频 SNR、仿真离线投影噪声 SNR、接收光功率、实际信号能量、时间槽和采样数。
连续特征没有可定义的比特 BER，记录为 null。原文只供发送端编码和离线评估，不传原始 UTF-8、不给接收端原文。
接收可用且 PSNR≥20dB、normalized WER≤.1 的比例是固定诊断质量目标，不能称独立语义成功率或 CRC。

## 上传与安装

把本次压缩包和同名 `.sha256` 一起上传到服务器的 `/data1/zw/WYH/multimodel/`。
请保留原始完整训练目录和三个 `last_delta.pth`，返回记录压缩包里没有大权重。

```bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_fso_jscc_eval_v1.tar.gz.sha256
test ! -e _v10_fso_jscc_eval_v1 && tar -xzf v10_fso_jscc_eval_v1.tar.gz
/home/zw/anaconda3/envs/torch/bin/python -B _v10_fso_jscc_eval_v1/install.py --project-root /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B _v10_fso_jscc_eval_v1/install.py --project-root /data1/zw/WYH/multimodel --verify-only
```

安装拒绝不匹配的旧依赖或已存在的不同版本，不自动覆盖。如果旧源码被修改，发回错误，不要通过关闭校验强行运行。
解压目录已存在时，上面的 `test` 会阻止再次解压；同一包可直接重新运行安装核验。

## 一条流程：预检 → 两图小验证 → 扩大验证 → 打包

使用新的输出目录，不复用旧实验目录。直接写解释器路径，避免空的 `$PYTHON_BIN`。

```bash
cd /data1/zw/WYH/multimodel
test ! -e experiment_results/v10_fso_jscc_eval_v1_01 && test ! -e experiment_results/v10_fso_jscc_eval_v1_01.log && nohup /home/zw/anaconda3/envs/torch/bin/python -B -u v10_extension/fso_jscc_eval_v1/launch.py pipeline experiment_results/v10_fso_jscc_eval_v1_01 --max-hours 6 > experiment_results/v10_fso_jscc_eval_v1_01.log 2>&1 < /dev/null &
```

默认读取：

```text
experiment_results/v10_fso_jscc_v1_01/v10_screen/
experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth
experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth
v10_extension/manifests/ln_v10_splits.json
COCO2017/
pretrained/sd-vae-ft-mse/
pretrained/clip-vit-base-patch32/
```

如果你的已完成 FSO 目录不是这个名字，在启动命令中加 `--source-run 实际的完整目录`。
不要给它训练返回包的解压目录，它不含三个权重。
其余基准权重也可用 `--checkpoint` / `--a-delta` 指向原文件；必须仍是原来的哈希。
物理 GPU 固定为 1，不影响别人的进程。只限制并结束本流程自己创建的子进程，不杀其他用户任务。
6 小时是全流程安全上限，不是预计耗时。两图小验证上限 15 分钟，成功后按两个引擎的实测速度显示预计扩大验证分钟数；加载和报告时间另计。
达到上限/异常会停止后续阶段，保留部分记录，不静默跳过失败。再次启动需新的 `_02` 输出和日志名称，不覆盖部分结果。

可先只检查或只跑小验证：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/fso_jscc_eval_v1/launch.py check
/home/zw/anaconda3/envs/torch/bin/python -B -u v10_extension/fso_jscc_eval_v1/launch.py smoke experiment_results/v10_fso_jscc_eval_v1_smoke_01 --max-hours 1
```

## 查看进度

```bash
tail -n 40 experiment_results/v10_fso_jscc_eval_v1_01.log
tail -n 20 experiment_results/v10_fso_jscc_eval_v1_01/v10_smoke.log
tail -n 20 experiment_results/v10_fso_jscc_eval_v1_01/v10_expanded.log
pgrep -af 'fso_jscc_eval_v1/(launch|run).py'
nvidia-smi -i 1
```

小验证尚未通过时 `v10_expanded.log` 不存在是正常的。正式日志 `EVAL … 31232 …` 显示累计已完成条数。
`v10_expanded/progress.json` 有按引擎实测速率投影的剩余时间；尚未计时的引擎和最终报告不计入该中间估计。

## 完成后发什么

```text
experiment_results/v10_fso_jscc_eval_v1_01/v10_fso_jscc_eval_v1_records.tar.gz
```

该包含全部逐样本记录、失败样本、原训练关键证据、冻结划分、前后权重/状态校验、旧结果重放、曲线、原图/四组恢复图、文本和星座图，不含模型/预训练权重和完整数据集。
本地打开 `v10_expanded/report.html` 查看图文对照；`summary.json` / `gain_table.csv` / `REPORT.md` 有数值与区间。
需要独立复核或重新打包时（只对已完成目录，使用新的包名）：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/fso_jscc_eval_v1/report.py experiment_results/v10_fso_jscc_eval_v1_01/v10_expanded --verify-only
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/fso_jscc_eval_v1/collect.py experiment_results/v10_fso_jscc_eval_v1_01/v10_expanded experiment_results/v10_fso_jscc_eval_v1_01/records_copy.tar.gz
```

## 结论边界

确认是否有可推广到新增验证图片的 FSO 训练收益，重点看主比较增益、区间、尾部恢复和失锁，不提前保证正收益。
图片 bootstrap 不是训练随机性区间，次要比较未多重校正。这不是最终独立测试、数字/JPEG 对比或联合/分别处理对比。
公共符号/采样/槽时长/峰值上限一致，不代表每帧实际能量严格相同；SNR 是仿真诊断，不是硬件测量。
不修改当前文字解码结构，因此扩大验证不会自动解决文字质量问题；保留真实差结果，不只展示高 SNR 好图。
