# DOCCI 独立图像/文本分支 V9

这一版先跑通两个私有分支，不训练 Common、Long-CLIP 或跨模态融合。它是后续联合实验的工程基线，不表示放弃图文共同恢复的目标，也不能用来证明联合增益。

## 变化与历史初始化

- 文本：原始 DOCCI 描述 → 冻结 BART encoder → 可训练32个64维语义槽 → 连续符号 → 可训练接收记忆 → 冻结 BART decoder 自回归恢复。完整输入512 tokens，不使用CLIP的77-token限制；不强制结束、不禁止重复。默认复用 V6 的正式 `P0_ce` 通信层，保留32槽和全部接收层，将已学习128输入位置保留，新输入位置偏置为零且可训练。V6当时32条短COCO验证描述平均WER约13.7%，不是DOCCI成绩。
- 图像：原始 DOCCI 图片完整视野 → 512×512等比例letterbox → 冻结VAE encoder → 可训练空间通信encoder/decoder → 冻结VAE decoder。只复用V8训练过的图像encoder/decoder，不导入失败的文本/Common权重。不是把128恢复图放大，也不使用扩散模型。
- 文本进程不加载图像/VAE/CLIP权重，图像进程不加载BART/CLIP权重。图片分支编码器没有描述输入。数据读取器会审计统一注释中的训练/开发分割，但另一模态不进入特征或网络。
- 两个分支质量独立验收；文字质量不合格也会继续图片分支。真实工程错误（非有限梯度、缺权重、坏检查点等）停止并保留日志，不伪装通过。
- 冻结骨干，但通信编码器和解码器均参与梯度学习；BART/VAE前向没有在训练阶段被错误包成整个 `no_grad`。原始信息只用于发送端、训练损失或事后评价，不作为接收端侧信息。

## 文件位置（沿用你服务器的现有文件）

项目 `/data1/zw/WYH/multimodel` 根目录中的 `docci_descriptions.jsonlines`、`docci_metadata.jsonlines`、`docci_images.tar.gz` 和 `longclip-B.pt` 不需要移动。已解压图像须在 `images_unpacked`、`docci_images` 或 `images` 下面。Long-CLIP文件这一轮不加载。

已有骨干为 `pretrained/bart-base` 和 `pretrained/sd-vae-ft-mse`。不联网、不安装/升级依赖；使用现有torch环境。GPU固定物理1，进程内cuda:0；先检查空闲≥16GiB，保留4GiB余量，不操作别人的进程。磁盘空闲≥8GiB。

默认历史检查点（同时需要其相邻的 `TRAIN_COMPLETE.json`）：

```
experiment_results/v10_clip_bart_joint_v6_03/v10_screen/text_prefit/clean/last.pth
experiment_results/v10_docci_longclip_joint_v8_root_01/v10_diagnostic/prefit/image/last.pth
```

已固定这两个检查点及其原始BART/VAE的SHA256。找不到时不会偷偷随机初始化。若你移动了相同检查点，可用 `--text-init /绝对路径/last.pth --text-init-sha256 SHA256` 和对应图像参数；旧文件不覆盖。确实要从头训练须明确 `--text-init fresh` / `--image-init fresh`，结果会注明与默认历史初始化不同。不要为通过校验随意修改校验值。

## 先运行这一轮，不直接启动大训练

新压缩包只含代码，以 `_v10_docci_modal_v9` 为解压目录。上传至项目根目录后：

```bash
cd /data1/zw/WYH/multimodel
sha256sum v10_docci_modal_v9.tar.gz
test ! -e _v10_docci_modal_v9 && tar -xzf v10_docci_modal_v9.tar.gz --keep-old-files
/home/zw/anaconda3/envs/torch/bin/python -B _v10_docci_modal_v9/install.py --project-root .
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_modal_v9/launch.py data-check --branch both
/home/zw/anaconda3/envs/torch/bin/python -B -u v10_extension/docci_modal_v9/launch.py diagnostic experiment_results/v10_docci_modal_v9_diag_01 --branch both
```

这轮会先自动执行服务器CPU契约测试和真实骨干/GPU检查：文本512形状；图像256与512形状、梯度、保存后回读。烟雾测试的权重不进入正式诊断，也不要求短暂8步内得到好成绩。

之后每条分支都有原生骨干参考、历史初始化在DOCCI上的表现、4个TRAIN样本小样本过拟合，以及8个固定qual_dev样本的实际恢复。文本过拟合1200次真实optimizer更新（batch4）；图片256次optimizer更新（microbatch1，累积2，共512个microsteps）。文本每200次更新保存真实自由生成的三控制诊断；评价不喂入原始目标前缀。

`DIAGNOSTIC_COMPLETE` 仅表示训练4样本的通信记忆通过诊断，**不表示开发集泛化已成功**。未达到恢复要求记为 `QUALITY_STOP`，失败文字、重复、容量停止、失败图片完整保留。每条分支均另存原生骨干参考，明确标注ORACLE、不经过传输。

仅有这轮结果后再决定是否运行 `screen`。`screen` 会重新从历史初始化出发，不能继承TRAIN4过拟合检查点：文本32 TRAIN、2000 optimizer更新；图片32 TRAIN、512 optimizer更新；各自在8个固定qual_dev上验收。若要只跑一条分支，使用 `--branch text` 或 `--branch image`，同时换一个全新输出目录。

`pipeline` 是明确请求后才运行的后续模式。其私有干净开发集通过后，把同一个干净检查点分别接着做等量的 `clean_continue` / `fso_continue`，文本各500更新、图片各256更新；数据顺序和学习率匹配。带噪训练使用Gamma-Gamma/指向误差符号级可微近似、25%干净锚点及TRAIN-only干净记忆/潜变量约束；不是通过真实完整DSP反向传播。之后在干净、weak_p06、moderate_p18、moderate_p21、strong_p18五条件下实际执行原V3光波形接收机，R=1，不重复24/50次；保留同步失败，不使用测试集调参。

## 时间、资源与结果

诊断加自动检查先预留 **15–45分钟**，这是从旧版少量实测速度推算的排期范围，尚未在你服务器实测。自由生成长描述可能成为主要开销，日志 `ETAtrain` 只估计当前训练循环、不含剩余评价/其他分支。默认4小时为安全总上限，不是预计耗时；若日志速度差异大，应据实际进度更新估计，不提高全局限制或抢占其他任务。

不要覆盖旧目录；若 `_01` 已存在就换 `_02`。查看当前进度：

```bash
tail -n 30 experiment_results/v10_docci_modal_v9_diag_01/v10_diagnostic_text.log
tail -n 30 experiment_results/v10_docci_modal_v9_diag_01/v10_diagnostic_image.log
```

先文本、后图片串行运行，所以图片日志出现较晚是正常的。结束会自动打包：

```
experiment_results/v10_docci_modal_v9_diag_01/v10_docci_modal_v9_records.tar.gz
```

把这个文件发回即可。包含两分支全部诊断、历史基线、实际图片/全文、接收帧、质量停止、原生参考与日志，不含预训练权重/训练检查点/原始高清数据。顶层 `index.html` 链接每条分支 `actual_recovery.html`。若只运行 `smoke`，结果仅为工程检查，不是恢复效果。

## 解释边界

正式诊断使用官方TRAIN和qual_dev；不使用test/qual_test模型输入。8条验证和一个训练初始化仅供方向筛查，不能作为论文泛化或多种子稳定性结论。门槛是事先指定的工程诊断：文本TRAIN WER≤0.2、开发集≤0.6，自然EOS且比清零/打乱记忆WER低至少0.1；图像PSNR≥18dB且比清零/打乱高至少1dB。18dB不是通用“成功恢复”定义；也不要求4个训练样本和验证表现一致。

这轮C=0。文本私有2048实数=1024复符号；512图像私有8192实数=4096复符号，256检查为1024复符号。真实光链路评价按各自短帧收费，不为缺少的模态发送大串零。AWG64GSa/s、4SPS=16GBaud、RRC滚降0.1；包含探测帧、导频、前导、滤波尾和guard。两模态资源不同，**不能直接宣称等资源联合/分别收益**；连续特征没有比特标签，BER/比特率记为未定义，不能编造传统比特压缩率。干净模式40dB只是公共标记，不是实测SNR。FSO参数沿用未硬件校准的模型；不声称实链路一致。

V8及所有旧源/数据/权重/结果都不修改；安装前检查源依赖，新增模块后对共享HANDOFF原字节备份再附加新条目。安装冲突不覆盖；旧版仍可按原命令运行。
