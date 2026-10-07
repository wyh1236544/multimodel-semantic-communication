# 光链路真实性 V3：接收机修复与短验证

状态：2026-10-07，新增候选版本。旧V2、权重、数据划分和实验结果不变。无新增训练，无原文旁路；本地NumPy改善不等于真实图文收益。证据见 [VERIFICATION.md](VERIFICATION.md)。

## 上传、解压与安装

将 v10_optical_realism_v3.tar.gz 及其 .sha256 上传到 /data1/zw/WYH/multimodel/。每一步成功后才进入下一步；暂存目录已存在时不删除或覆盖，先检查来源。

~~~bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_optical_realism_v3.tar.gz.sha256
test ! -e _v10_optical_realism_v3 && test ! -L _v10_optical_realism_v3 && tar -xzf v10_optical_realism_v3.tar.gz
/home/zw/anaconda3/envs/torch/bin/python -B _v10_optical_realism_v3/install.py --project-root .
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/optical_realism_v3/install.py --project-root . --verify-only
~~~

安装器核对全部源码及原有依赖哈希、链接与冲突后，仅新增 v10_extension/optical_realism_v3。共享HANDOFF按字节备份到deployment_records后再追加；旧模块不替换。需已安装V2、final_compare_v1及其原有依赖，版本不同会停止，不跳过检查。

源码包只含本版代码/配置/文档，不含权重/数据/结果/凭据；确定性USTAR不携带macOS扩展属性。不重装服务器环境，不从GitHub挑几个文件覆盖V2。

## 第一次运行：GPU1，不训练

先只读查看GPU；不停止他人进程。运行器要求至少8GiB空闲显存/磁盘，但不保证共享GPU的全程资源。

~~~bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
cd /data1/zw/WYH/multimodel
CUDA_VISIBLE_DEVICES=1 /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/optical_realism_v3/launch.py pipeline \
  experiment_results/v10_realism_v3_quick_20261007_01 --max-hours 1
~~~

已有目录或日志时换成_02，不删除旧结果。直接使用完整解释器路径，避免为空的PYTHON_BIN。

流水线：合同及Torch ReplayFrame测试 → 15个B2B功能检查 → 24个同ADC观测的V2/V3合成回放 → 2张真实validation冒烟 → 两种数字策略各8张validation对照 → 独立验收、报告、打包。

每策略96行：8图×3工况×1通道抽样×3路线+24行无信道参考。连续结果在两策略中重复，不是16张独立图片。沿用现有混合训练基础权重、A增量、COCO/VAE/CLIP与冻结划分，核对精确身份。不新增训练种子、不读test、不自动扩到32×9×3。

依据V2每份8图评估约3分钟，先预留10–20分钟；模型加载、CPU和GPU占用可能改变时间，以progress.json实测为准。1小时是停止上限，不是预测。

## 交回结果与进度

完成后只发：

experiment_results/v10_realism_v3_quick_20261007_01/v10_optical_realism_v3_records.tar.gz

~~~bash
tail -n 12 experiment_results/v10_realism_v3_quick_20261007_01/smoke.log
tail -n 12 experiment_results/v10_realism_v3_quick_20261007_01/qpsk_conv.log
tail -n 12 experiment_results/v10_realism_v3_quick_20261007_01/16qam_repeat2.log
~~~

未开始阶段没有日志是正常的。失败时发对应日志最后80行。结果包含所有失败、实际图片/原文/恢复文本、拒收前后星座、BER、锁定、SNR、时间/光能量、配置/源码/权重身份；不传权重或数据集。

## 改动与物理边界

1. 接收同步：前导偶数导频做有界相干CFO匹配，默认±250MHz、约4MHz网格再局部细化；最多32个公开导频块联合估计时钟，不逐块追噪声。复增益平滑后跟踪相位，奇数导频验证。接收器不读真实CFO/时钟/衰落/发送特征/离线SNR，不对连续载荷作QAM判决。
2. 同步分数：V3的0.1是128个奇数验证导频的整段相干相关，V2的0.2是另一种短块相关，定义不同，不是简单降低同一门限。−6dB残差、相位及捕获边界检查继续保留；默认时钟范围±200ppm，无足够跨度时报告未拟合。有限导频不是完备周跳检测。
3. 连续可靠性：取导频残差/R、重复分半均值差换算方差、未除R的导频块平均残差三者最大值。三者均来自接收观测；后者含有限块噪声，刻意保守。用于LMMSE式收缩/模型可靠性，失锁为0；不是实测误差上界，也不能检测所有一致错误。发送特征MSE只作解码后离线诊断。
4. 16QAM：V2相邻重复比特只覆盖4点；V3完整编码块发送两遍，每4位Gray映射，再按对应编码位累加两遍LLR作软Viterbi。16点均可到达，重复比特、终止码/填充、导频/尾部/保护间隔全部计费。仍是K7 1/2卷积码，不是LDPC/Polar或最优数字方案。
5. 报告：修复JSON尾部字面反斜杠n、Markdown字面换行及生效preset记录；验收真正解析JSON。分别记录探测/载荷/全部突发锁定，以及锁定但特征MSE>1的数量。星座保留拒收前均衡观测和拒收后交付点。
6. SNR：waveform_noise_only_snr_db是波形光学/散粒/热噪声分项比；physical_symbol_snr_db及rx_selected_projection_noise_snr_db是接收器选择后的信号投影噪声比，失同步也会降低分子；导频残差和拒收前EVM另列。均是仿真，不是仪器实测。

物理仍为64GSa/s、4SPS、16GBd、RRC滚降0.1、IQ-MZM/相干探测及既有FSO/ASE模型。4µs上限、公共激光工作点和峰值包络相同，不代表实际等平均功率或等消耗能量。连续仍最多24次，同慢衰落块内重复不是免费分集。CPU墙钟不是光学传输/硬件DSP延迟。

神经网络未改：A的已观察无信道文字WER约55%的瓶颈不会因此自动解决。这不是已经为该光链路优化过的DeepJSCC。

## 后续选项：先不要全部启动

只跑合成接收回放，无GPU/权重：

~~~bash
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/optical_realism_v3/launch.py diagnostic \
  experiment_results/v10_realism_v3_receiver_only_01 --max-hours 0.25
~~~

主短验证通过后才考虑ideal及少量中间光功率点：

~~~bash
CUDA_VISIBLE_DEVICES=1 /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/optical_realism_v3/launch.py quick \
  experiment_results/v10_realism_v3_ideal_01 --preset ideal --max-hours 0.5

CUDA_VISIBLE_DEVICES=1 /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/optical_realism_v3/launch.py quick \
  experiment_results/v10_realism_v3_midpoints_01 \
  --condition-ids moderate_p21 moderate_p18 moderate_p15 --max-hours 0.5
~~~

中间点是−21/−18/−15dBm公共校准参考光功率，不是预设SNR；仍仅8图、1通道抽样。先看good是否退化、弱链路PSNR/失败率是否改善、错误放行是否减少、CRC/BER/预算是否一致。全部失败计分；不能只挑成功图。若仍无质量—资源收益，应保留负结果再缩减连续主张，不能隐藏原文旁路、改评价门限或删除失败。

