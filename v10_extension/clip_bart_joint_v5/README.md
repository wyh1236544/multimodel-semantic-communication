# CLIP+BART联合模块V5：先保住文字，再验证图文融合

状态：新增候选训练代码，不是已验证联合优越性。历史V1–V4、CLIP/VAE/B4、
数据、划分、权重、结果全部保留。新增源码只在本模块，不改旧模型和环境。
本地无Torch/预训练，真实GPU、学习与效果必须由服务器smoke/训练验证。

## 为什么这样改

前BART+CLIP V1与COCO单模态V2都停在文字干净诊断；旧V3的T压缩接口仍差。
V4两组在同8句TRAIN上实际无约束恢复8/8、WER0，说明查询压缩→固定语义
记忆接口可以学通；不证明泛化。V5沿用V4接口，不退回四位置拼接/128位置
展开，也不使用TRAIN8拟合权重当正式初始化。历史依据见[SOURCES.md](SOURCES.md)。

## 结构

同一句COCO原caption有两种表示，不是偷偷发送两份原文：

```text
图片 → 冻结VAE/B4图像编码 → I224×16 ─────────────────────┐
                  └图像局部特征＋冻结CLIP图像语义 ┐       │
文字 → 冻结CLIP文本语义 ─────────────────────────→公共C32×16 → 信道
文字 → 冻结BART编码 → V4查询压缩 → T32×64 ────────────────┘

收到C/I/T → C辅助图像接收层＋固定B4/VAE → 实际图片
          → C辅助文本语义记忆＋固定BART → 实际文字
```

学习的是T codec、公共C编码/辅助任务、图像TX/RX小适配器及两侧融合层。
BART、CLIP、VAE、历史B4全部冻结。C监督同时对齐CLIP图像/文本语义，并预测
图像4×4粗latent；训练teacher/原图/原文只用于损失，不是接收输入。
初始C→文本/图片残差为0，保持原私有接口；干净训练前200forward冻结T，
之后T以2e-6小LR联合微调，并约束到冻结泛化文本codec的记忆。没有原128词
逐位置MSE。公开C两半分别单位功率归一化，T每个槽单位功率，I分组单位功率。

接收端只输入实际载荷、锁定状态与导频估计SNR/可靠性；BART记忆固定32槽，
输出EOS决定长度，没有免费源长度/mask/UTF8旁路。当前图片由接收C/I恢复，
不是把源文本喂给图片生成器；不加载扩散，不运行ControlNet或原文提示生成。
C/I/T不是已证明互信息完全解耦的变量，不这样宣称。

## 三组同预算独立训练

|组|编码与接收规则|目的|
|---|---|---|
|J0_separate|C前16只能看图片，后16只能看CLIP文字；图像RX只用前半、文字RX只用后半|神经分开处理对照|
|J1_joint|C各槽可以看图文，两个任务可以看全部接收C|联合处理候选|
|J2_no_clip_text|J1但源CLIP文本向量置零；BART文字T和CLIP图像仍保留|CLIP文字输入作用|

三组注册层、初始张量、数据顺序、训练步数、损失、LR、总符号预算相同。
J2只删CLIP文本SOURCE，仍保留同样CLIP训练teacher，避免同时改变监督目标。
相同登记参数不等于实际梯度参数/FLOPs相同，均单列审计。
J0是共享同一物理帧的神经块隔离，不是多收两次帧头的“两包传输”，不是
搜索过最优分配的最强单模态系统。传输排布、光链路与同步均保持一致。

总C512+I3584+T2048=6144实数=**3072复符号**，R=1。
比旧2304更多，不能直接说比旧版更省资源；V5三组互相比才是同资源筛查。
两组CLIP/BART文字表示是神经输入，不额外发送原文；没有FEC/数字UTF8旁路。
导频/probe/preamble/tails/guards全计费，重新生成完整3072帧，不切旧2304bank。
相同slot/峰值包络不是严格相同实测光能量；不伪造analog BER/CRC/字节压缩比。

## 训练顺序与停止条件

1. 核验已有V4正式诊断目录，两组通过且不是smoke；不加载其拟合权重。
2. 新初始化P0纯CE文本codec：1024TRAIN caption，batch4×3000optimizer更新，
   冻结BART；32validation贪心/零/真实外部错配记忆。WER<=.60、零/错配恶化
   >=.10、全部自然EOS才进入联合阶段；smoke质量绕过明确标注、不能初始化
   正式训练。门禁未通过时保留所有文字，标text_prefit_stop，不放松阈值。
3. 三组从相同正式泛化文本codec＋相同新公共/融合层初始值开始，256配对
   图文×4clean epochs，batch1/accum2=各1024forward/512optimizer；前200步
   T冻结，后续2e-6。C/image/fusion LR1e-4。损失CE＋latent1＋pixel.1＋
   共同语义对齐.1＋C粗latent.05＋文本记忆anchor.1。固定最后一步，不按
   validation选最优checkpoint。每组都报告clean实际图片/文字，即使质量差。
4. `screen`到此结束：每组评估clean+原6种完整FSO条件，但本轮只有干净训练。
5. 可选`pipeline`：每组自身clean文字验证过门槛后，**同一个clean权重**分叉
   clean_continue与FSO，各256×2epochs、512forward/256updates。FSO每4步
   1个clean anchor，其余moderate/strong与-5/0/5/10/20dB轮换；3e-5融合LR、
   T仍2e-6。两分支步数/损失/顺序相同，只有信道不同。不自动扩大样本预算。

FSO训练代理沿用V4的Gamma-Gamma光强→sqrt场增益、指向损失、相位、固定
衰落前噪声、32导频估计与均衡。SNR为名义符号量，不是光功率推导的SNR。
代理不包含DAC/MZM、带宽、同步拒绝或相关天气，不声称匹配完整波形。
最终评价独立用原64GSa/s/4SPS/16GBaud/RRC.1/V3同步和实际光功率配置，
保留失锁、全部误差、实际光功率/能量/导频SNR/时长/星座。

主比较预先定义J1-J0同阶段正常接收的PSNR与WER；同时看SSIM/失锁和资源。
J1另外保存C置零、关闭C融合、C外部错配的同接收帧干预，以及clean条件
发送端错配文字（CLIP与BART都用供体文字，图像仍为自己）。这些干预是
依赖性诊断，不是重新训练的最优无C方法，不把借用供体记忆当实际免费传输。
不删失败图、不只挑好看样例。固定前4个验证ID展示全部条件图片，所有文字
都保存到actual_image_text.html。PSNR含失锁黑图，不能用成功子集掩盖失败。

## 上传与安装

下载发布包与.sha256到本机桌面，在Mac终端上传，替换SSH地址：

```bash
scp /Users/a1-6/Desktop/v10_clip_bart_joint_v5.tar.gz /Users/a1-6/Desktop/v10_clip_bart_joint_v5.tar.gz.sha256 zw@服务器IP:/data1/zw/WYH/multimodel/
```

服务器分段执行，任一步出错先停，不删除已有冲突：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_clip_bart_joint_v5.tar.gz.sha256
test ! -e _v10_clip_bart_joint_v5 && tar -xzf v10_clip_bart_joint_v5.tar.gz
"$PYTHON_BIN" -B _v10_clip_bart_joint_v5/install.py --project-root /data1/zw/WYH/multimodel
```

包是纯源码USTAR，不含数据或权重，不需要下载新模型或安装依赖。
安装器校验V4及历史依赖，再新增V5，HANDOFF按原字节备份后追加，不覆盖。
相同安装可重验；不同内容拒绝。Py3.8/Torch1.12/Transformers4.30.2/
Diffusers.20.2现有环境保持，不安装PEFT/einops或升级库。

## 第一轮建议：先clean三组，GPU1

```bash
"$PYTHON_BIN" -B v10_extension/clip_bart_joint_v5/launch.py data-check
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used,memory.free --format=csv
"$PYTHON_BIN" -B v10_extension/clip_bart_joint_v5/launch.py screen experiment_results/v10_clip_bart_joint_v5_01 --max-hours 4
```

自动先CPU合同、10个新Torch/HF合同，再真实小smoke，再正式预训练/联合筛查。
至少16GiB空闲/运行保留4GiB，磁盘12GiB，不足停止，不杀别人进程。
预计smoke5–15分钟，首轮clean筛查30–90分钟，**未经V5实测**；V4已测
3000文本updates按旧速度约10分钟，但新数据/图片/B4/自回归/显卡竞争会改变
实际时长。4小时是总停止上限，不是ETA。阶段日志按实测速度每20步更新。

需要断SSH后继续时，只选后台命令一次，不与前台重复启动：

```bash
mkdir -p experiment_results
test ! -e experiment_results/v10_clip_bart_joint_v5_01 && test ! -e experiment_results/v10_clip_bart_joint_v5_01_launch.log && nohup /home/zw/anaconda3/envs/torch/bin/python -B -u v10_extension/clip_bart_joint_v5/launch.py screen experiment_results/v10_clip_bart_joint_v5_01 --max-hours 4 > experiment_results/v10_clip_bart_joint_v5_01_launch.log 2>&1 &
```

查看进度：

```bash
tail -n 60 experiment_results/v10_clip_bart_joint_v5_01/v10_smoke.log
tail -n 60 experiment_results/v10_clip_bart_joint_v5_01/v10_screen.log
/home/zw/anaconda3/envs/torch/bin/python -m json.tool experiment_results/v10_clip_bart_joint_v5_01/v10_screen/progress.json
```

完成或质量门禁未通过，都发：

```text
experiment_results/v10_clip_bart_joint_v5_01/v10_clip_bart_joint_v5_records.tar.gz
```

异常未生成包则发当前日志、FAILED.json、learning_audit.json。
SCREEN_COMPLETE只表示流程跑完，不是联合有效的证明。失败不自动重试，不
覆盖01；新尝试改02。大权重last.pth留服务器。此版不支持精确断点续跑。

默认只读旧输入路径在launch.py中。V4实际目录不同用`--text-screen`指向其
**v10_screen**，不是外层父目录；B4等旧输入可用--source-run/--legacy-run/
--checkpoint/--a-delta/--bart-path/--vae-path/--text-model-path明确覆盖。

首轮结果确认后，已有代码支持包含FSO的完整短筛查（新目录，重新预训练，
不是续训；粗估1–3小时未实测）：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/clip_bart_joint_v5/launch.py pipeline experiment_results/v10_clip_bart_joint_v5_fso_01 --max-hours 4
```

不访问test/calibration业务数据，不扫训练种子，不混入长LN/新数据集。
当前只比较固定COCO同caption的CLIP与BART表征。后续长/短caption、缩小T
容量、数字/JPEG同资源强基线、最终独立test/多种子与真实硬件另行定义。
本轮不保证联合收益、连续优势或投稿；检查与效果边界见[VERIFICATION.md](VERIFICATION.md)。
