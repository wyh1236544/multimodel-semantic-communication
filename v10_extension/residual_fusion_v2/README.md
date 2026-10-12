# V10：受控接收端图文残差融合 V2

状态：候选诊断实现，尚未测得真实恢复收益。旧版源码、权重、结果不覆盖。
从完成的 `joint_jscc_arch_v1` **B4分别处理组**初始化；不是接着B3继续混合。

## 本次做什么、没做什么

1. **文字定位**：各用32条训练文本、1000次更新，分别训练两个完全相同初始化的AR解码器。D0看压缩前干净文本特征（仅诊断的源特征oracle，不是部署成绩）；D1看固定B4经过T32压缩的干净接收特征。D2让同一个已训练D1头读取全波形FSO后的T32，不再训练。记录真实greedy输出、EOS/容量停机、文字WER及原文/覆盖参考。
2. **质量门禁**：正式D0、D1的覆盖文本WER都须不高于10%。这是本版工程停止条件，不是保证泛化的准确率阈值。未通过即输出 `DIAGNOSTIC_COMPLETE.json` 和 `diagnostic_stop` 报告；不启动两组正式融合训练，不删记录、不放宽阈值。训练样例过拟合通过也不代表正式文字恢复已改善。
3. **小残差对照**：R0只读取接收图像记忆，R1读取实际接收到的文本C16+T32记忆，两组网络形状、初始化、参数数量、优化步数和增强调度一致。每组49,505可训练参数；B4、CLIP、VAE全部冻结。残差出口零初始化，接收失败门控为0；关闭融合、零文字都精确回到B4。门控不读取真实SNR/原文/天气真值，也不保证训练后不会退化。
4. **训练**：每组1000图文对，2轮干净+2轮FSO，4000前向、2000优化更新。drop10%、错配10%为配对记忆增强；R0同样处理图像侧记忆，R1处理文字侧记忆。梯度只进入残差模块，经冻结VAE解码器传播图片损失；不加入Common辅助损失，不优化文字主头，不需要PCGrad来修复本阶段不存在的共享任务梯度。
5. **同一接收帧比较**：完整V3波形每个图文/条件只运行一次；B4、R0、R1及关闭/零文字/不同图真实接收文本的干预全部复用同一接收张量。首次可以在接收端对照中严格保持同一实际时间、能量和损伤。错配是离线接收端因果干预，不是一次真实多图传输协议。

**正式文字输出仍来自冻结B4。诊断头不替代它，R0/R1不保证文字WER下降。**
本次先回答“正确的已接收文字token是否额外改善图片”，不是已经完成发送端联合编码，也不是完整文字解决方案。发送端联合Common、文字头泛化训练、全预算图像单模态/数字比较留给门禁通过且收益成立后的独立阶段。

保持连续R=1、C32/I224/T32、16实维、2304复载荷；原文UTF8旁路0字节。
保留64GSa/s、4SPS、16GBaud、RRC0.1及原FSO湍流、指向、功率链路。
最终成绩用完整波形，训练用已校准代理；固定B4实际载荷先做新噪声全波形holdout适用性检查。
失败帧不删除，不捏造模拟连续符号的BER。保存实际功率、SNR、发射能量、时长、PAPR、图片/文字和星座图。
clean记录40dB只是无噪声控制标签，不是有限实测光链路SNR。

## 上传、校验、安装

将代码包和校验文件上传到已有服务器 `/data1/zw/WYH/multimodel/`，不要新建并列项目。
在本地终端、代码包所在目录，可将已有SSH地址替换到下面命令；也可使用当前连接的上传工具。

```bash
scp v10_residual_fusion_v2.tar.gz v10_residual_fusion_v2.tar.gz.sha256 zw@你的服务器地址:/data1/zw/WYH/multimodel/
```

以下在服务器执行：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_residual_fusion_v2.tar.gz.sha256
test ! -e _v10_residual_fusion_v2 && tar -xzf v10_residual_fusion_v2.tar.gz
"$PYTHON_BIN" -B _v10_residual_fusion_v2/install.py --project-root .
"$PYTHON_BIN" -B v10_extension/residual_fusion_v2/install.py --project-root . --verify-only
```

安装器先核对历史依赖及全部新源码；冲突拒绝覆盖。共享HANDOFF只在保存原字节备份后追加。
若 `_v10_residual_fusion_v2` 已存在，先核对文件/哈希，不能直接覆盖旧源码。

## 启动：仍用GPU1

启动前检查可用显存；不足16GiB就等待，不停止其他人的进程。

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
nvidia-smi -i 1 --query-gpu=name,memory.total,memory.used,memory.free --format=csv
(
  set -C
  nohup "$PYTHON_BIN" -B -u v10_extension/residual_fusion_v2/launch.py pipeline \
    experiment_results/v10_residual_fusion_v2_01 --max-hours 3 \
    > experiment_results/v10_residual_fusion_v2_01.launch.log 2>&1 < /dev/null &
  echo $!
)
```

CPU/真实Torch合同 →8对短预检 →正式32条文字定位与门禁 →若通过则两组融合训练 →同接收帧完整波形评估 →独立验收 →自动收集。
短预检只训练2条文本各8步，不要求达到正式质量门禁；报告明确标记“未验证质量”。它只验证执行/梯度/文件接口，不能冒充正式通过。

必须保留服务器的完整旧权重，默认依赖：

```text
experiment_results/v10_joint_jscc_arch_v1_01/v10_screen/train_B4_separate/fso_delta.pth
experiment_results/v10_fso_fluctuating_v2_01/v10_screen/train_B2_fso_ft/last_delta.pth
experiment_results/v10_fso_fluctuating_v2_01/calibration/
experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth
experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth
```

若完成的B4目录不是 `_01`，在启动命令增加 `--source-run experiment_results/实际目录/v10_screen`。
旧V2 B2位置改变用 `--legacy-run`，bank改变用 `--calibration`。不能指向只有小记录包、没有权重的目录；身份不一致时程序会拒绝。
输出名已存在时用 `_02` 等新名字，同时更改外层日志名。没有精确优化器/RNG续训协议，不自动重试或无缝续训。

## 进度、时间、发回文件

```bash
tail -n 40 experiment_results/v10_residual_fusion_v2_01.launch.log
tail -n 40 experiment_results/v10_residual_fusion_v2_01/v10_smoke.log
tail -n 40 experiment_results/v10_residual_fusion_v2_01/v10_screen.log
pgrep -af 'residual_fusion_v2/(launch|run).py'
```

开始时只有预检日志是正常的；正式诊断会显示 `DIAG ... teacherCE`，随后显示实际生成的 `TEXT DIAG gate`。
融合训练每100步更新 `v10_screen/progress.json`。加载/诊断阶段尚无此文件，不代表卡死。
按30–90分钟规划整轮小筛选，正式门禁失败会更早停止；这不是已实测新版本时间。
3小时总预算、单段1小时和45分钟预检均为停止预算。GPU真实预检及前100–200步修正估计。

无论完整跑完还是文字门禁停止，自动收集：

```text
experiment_results/v10_residual_fusion_v2_01/v10_residual_fusion_v2_records.tar.gz
```

把这个包发回来即可。若运行异常而没有包，发外层日志、最新 `v10_smoke.log`/`v10_screen.log` 和 `FAILED.json`。
大权重、数据、bank留在服务器。报告入口 `v10_screen/report.html`，汇总 `summary.json`。
主验证32对、clean+6物理条件、6变体=1344结果行；只有192个有噪物理帧、384个probe/features事件。
没有训练种子扫描，每条件1个固定天气路径；按图片聚类区间不包含训练/天气总体变异。历史validation不是最终独立test。

## 依据与边界

- [DeepSC](https://arxiv.org/html/2006.10685v3)：参考真实文本重建与语义/逐词指标分离；不声称复现其大语料训练、MI目标或理想CSI结果。
- [跨模态门控研究](https://arxiv.org/abs/2208.11893)：借鉴受控信息融合；原任务为情感分析，不是本项目FSO有效性的证明。
- [Gradient Surgery](https://arxiv.org/abs/2001.06782)：提醒检查共享任务冲突；本阶段冻结主干、只优化图片残差，没有实现/声称需要PCGrad。

本机无Torch，不能从本地语法/NumPy检查宣称服务器模型有效。服务器CPU Torch测试和真预检是强制步骤。
