# V6：延长文字训练、让Common真正参与恢复

状态：新增候选代码，非已验证收益。V5/fix1及所有旧权重/数据/划分/结果保留。
不改现有Python3.8/Torch1.12/Transformers4.30.2/Diffusers0.20.2环境。

## 实际改动与边界

1. 校验你完成的V5_02正式screen与text_prefit/clean/last.pth的身份及SHA。
   只加载其P0/3000文本codec，绝不加载8句诊断或smoke权重。
   分两段各3000更新，从3000继续到6000、9000；每段重新创建AdamW、LR和
   sampler，不是精确断点续跑。与V5同1024原COCO描述、batch4、seed42。
   保存全部32验证句在三里程碑的贪心/零/错配/beam4输出，固定最后9000
   权重，不按验证选优。原始BART未压缩memory只作标注清楚的离线oracle，
   不进入真实接收，不计作通信成绩。9000门槛仍WER≤.60、依赖差≥.10、自然EOS。
2. 图像局部特征+CLIP图像先池化16槽，CLIP文本全局向量独立池化16槽后融合。
   文本16槽并不意味着额外16词或更多原始信息，骨干仍冻结，C仍32×16。
3. 每个arm先256对×2轮Common warmup：caption/coarse/alignment；私有codec
   与正常融合冻结。再256对×8轮联合：同预算的四步全量/遮图/遮文/两者遮挡
   策略，遮挡25%的私有槽，C不遮挡。遮挡发生在接收特征，是训练正则，不是
   新FSO信道、免费擦除标记或带宽节省。没有改真实发送帧。Common描述头输出
   固定32×768记忆辅助BART；其零初始化门控接入正常文字记忆。粗latent4×4
   的门控差值残差接入图像，保留私有高频细节。门控初始0，warmup不更新。
4. J0同帧神经分开、J1图文联合、J2删源CLIP文本（保留BART T与同CLIP监督）；
   三组同登记参数/初始张量/数据/更新/损失/掩码/物理资源。登记参数不等于
   实际梯度数/FLOPs相同。Common-only/错配/置零仍按完整3072帧收费，非新省码率。
5. 默认screen只做clean训练+原6种完整FSO评估。另报告共同描述和固定32对
   验证特征遮挡的恢复，保留所有错误，不将这个人为遮挡当实链路鲁棒性。
   pipeline额外做自身clean通过后的同权重clean_continue/FSO配对，各2轮；
   两分支同掩码/数据/步数，只有符号代理噪声不同。代理不含完整波形/同步/
   时间相关天气/DAC/MZM；最终完整V3光链路评估保持原参数。
6. generate.py单独运行，不加载扩散到主训练：冻结现有完整SD，只训练小型
   C+估计SNR→接收文字条件残差适配器。64 TRAIN实际接收条件（clean+代理噪声）
   监督到冻结SD编码的原文teacher；原文/长度仅用于TRAIN损失，不是RX。
   这是文本嵌入回归，不是论文ControlNet/DiT/扩散去噪训练，不承诺图像改善。
   固定首4验证ID×所有条件输出全部候选，失锁不生成，不选最好seed/图。
   对照：直接重建、仅图、RX文字、RX文字+CSI但C置零、RX文字+C+CSI、错配C。
   所有生成条件用同seed/strength/步数，不增加实际通信载荷，但增加计算时延。
   原图只在推理完成后用于评价；所有图片/文字可视化。生成指标为PNG上的
   PSNR/SSIM；LPIPS/FID未安装/验证而显式为null，不假装感知收益已证实。
   没有完整SD权重时先跑主训练；sd-vae-ft-mse只有VAE，不能用于生成。

## 上传解压

下载包与.sha256到Mac桌面，替换SSH地址：

```bash
scp /Users/a1-6/Desktop/v10_clip_bart_joint_v6.tar.gz /Users/a1-6/Desktop/v10_clip_bart_joint_v6.tar.gz.sha256 zw@服务器IP:/data1/zw/WYH/multimodel/
```

服务器逐条执行；任何一步出错先停，不删除冲突：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_clip_bart_joint_v6.tar.gz.sha256
test ! -e _v10_clip_bart_joint_v6 && tar -xzf v10_clip_bart_joint_v6.tar.gz
"$PYTHON_BIN" -B _v10_clip_bart_joint_v6/install.py --project-root /data1/zw/WYH/multimodel
"$PYTHON_BIN" -B v10_extension/clip_bart_joint_v6/launch.py data-check
"$PYTHON_BIN" -B v10_extension/clip_bart_joint_v6/launch.py screen experiment_results/v10_clip_bart_joint_v6_01 --max-hours 4
```

固定GPU1。自动CPU合同→26 Torch/HF toy合同→真实saved模型smoke→正式screen。
主训练启动至少16GiB显存/12GiB磁盘，保留4GiB。不杀别人进程、不联网/升级库。
已测V5文本3000updates约4.6分钟；V6新增6000updates粗估10分钟，但额外验证、
Common/双BART损失、联合8轮和光链路评估成本增加。预计smoke5–20分钟、正式
screen60–150分钟，**尚未V6实测**；4h是上限不是ETA。无精确断点续训。
默认parent-screen=experiment_results/v10_clip_bart_joint_v5_02/v10_screen；
实际路径不同用--parent-screen覆盖，其余旧输入参数同V5入口，不覆盖旧目录。

后台运行时只选以下命令一次，不与前台重复启动：

```bash
mkdir -p experiment_results
test ! -e experiment_results/v10_clip_bart_joint_v6_01 && test ! -e experiment_results/v10_clip_bart_joint_v6_01_launch.log && nohup /home/zw/anaconda3/envs/torch/bin/python -B -u v10_extension/clip_bart_joint_v6/launch.py screen experiment_results/v10_clip_bart_joint_v6_01 --max-hours 4 > experiment_results/v10_clip_bart_joint_v6_01_launch.log 2>&1 &
```

进度与结果：

```bash
tail -n 60 experiment_results/v10_clip_bart_joint_v6_01/v10_smoke.log
tail -n 60 experiment_results/v10_clip_bart_joint_v6_01/v10_screen.log
```

发experiment_results/v10_clip_bart_joint_v6_01/v10_clip_bart_joint_v6_records.tar.gz；
质量门禁失败也发，异常未出包则发当前log/FAILED/learning_audit。大权重留服务器。
报告包含三里程碑文本、Common-only描述、正常/干预/人为遮挡图片文字、光功率/
估计SNR/能量/资源与失锁，不把SCREEN_COMPLETE当收益证明。所有输入只用TRAIN/
validation，不访问test/calibration；反复查看validation不能作为独立test结论。

## 后续可选阶段（不要现在重复启动）

带噪完整pipeline使用新目录，重跑前段再分叉，不是接着screen目录精确续跑：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/clip_bart_joint_v6/launch.py pipeline experiment_results/v10_clip_bart_joint_v6_fso_01 --max-hours 6
```

生成实验需要完整旧SD目录，下面占位路径必须替换。GPU1至少12GiB/保留4GiB；
不自动下载。主screen结束后，独立运行（若pipeline且J1实际训了FSO，用对应
screen目录和--phase fso）：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/clip_bart_joint_v6/generate.py --screen-run experiment_results/v10_clip_bart_joint_v6_01/v10_screen --diffusion-model-path /data1/zw/WYH/multimodel/pretrained/你的完整SD目录 --output-dir experiment_results/v10_clip_bart_joint_v6_generation_01 --max-hours 1
```

生成粗估10–60分钟未实测，1h是停止上限。发生成目录内
v10_clip_bart_joint_v6_generation_records.tar.gz；包含actual_generation.html及全部
候选。源权重文件/文献引用见SOURCES.md。不能从漂亮图片推出逐像素准确或无损。

## 本地验证边界

42新非Torch合同（10策略+32报告，含完整clean/paired-FSO合成矩阵、阶段预算/
Common遮挡/原文泄漏拒绝）；原V5/V4回归另跑。Python3.8语法、入口help和
包的实际还原/安装/重复/冲突/依赖漂移/符号链接/交接字节备份检查通过。
本地缺Torch/预训练/COCO，26 Torch/HF tests、GPU、BART/CLIP/VAE/SD真实运行、
学习/显存/效果均未执行，服务器预检必做。合成合同不是网络性能证据。
