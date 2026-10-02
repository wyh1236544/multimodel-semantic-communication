# V10 Optical Link V1：物理参数驱动的相干FSO

状态：新增隔离模块，2026-10-02。本地NumPy物理/协议验证与服务器真实模型验证分开记录，见[VERIFICATION.md](VERIFICATION.md)。不覆盖历史源码、权重、结果或自适应分支，不安装/升级服务器依赖。不是OptiSystem运行结果或硬件实测。

## 目标和边界

保留用户选定A的联合图文编码、C32/I224/T32×16、8位固定范围量化与整对象CRC擦除。使用既有mixed seed42完整V10基权重及已训练A的`last_delta.pth`，严格核对二者SHA与划分。**无重新训练、无原文UTF8旁路、无测试集选模、无扩散生成。** A仍有联合图文Transformer，不是图像单模态网络。禁用的只是A原有的两个显式文字→图像引导模块。

```text
A联合图文特征 → 固定量化/CRC/组帧/FEC/QAM → RRC/DAC
→ CW激光/IQ-MZM → TX EDFA → 发射光学损耗
→ 大气衰减/Gamma-Gamma/指向与孔径积分 → RX损耗/耦合
→ 可选RX EDFA/光滤波 → LO/90°混频/四PD平衡探测
→ 电滤波/TIA/ADC → 导频同步/估计/均衡 → QAM/FEC/CRC
→ A图片与文字解码
```

新物理模式**没有预设SNR参数**。`run.py`给旧模型接口的0只是被OpticalFrame忽略的兼容占位；不会进入噪声生成或神经门控。旧归一化SNR模式完全保留。神经门控只收到独立探测导频估计，沿用旧网络[-40,40]dB范围截取；模拟器真值与源数据仅供离线诊断。

## 配置：optical_config.json

完整`base`＋各`cases`局部覆盖。未知拼写、缺失参数、非有限数、冲突模式会拒绝，不猜测参数。`parameter_status=assumed_not_hardware_calibrated`为默认。只有全部输入确实来自测量时才用`measured_parameter_inputs`；这也不把仿真变成硬件实测。

建议复制为新的项目内配置，例如`v10_extension/manifests/optical_custom_01.json`，然后传入`--optical-config`，保留发行默认文件不变。不要根据图像好坏改参数来挑结果。

| 组 | 单位与合同 |
|---|---|
| waveform | Fs Hz；整数sps；RRC滚降0..1、偶数跨度；DAC位数；每包字节与保护样点；数字policy |
| laser | 波长m；CW输入功率dBm；Lorentz/Wiener相位线宽Hz |
| modulator | IQ-MZM或小信号linear_reference；额外插损dB；Vpi/峰值驱动/偏置V；I/Q相位偏差rad；假定3dB电带宽Hz |
| tx_edfa/rx_edfa | enable；gain或total_power控制；最大增益dB；目标和最大总输出dBm；NF dB |
| path | 距离m；大气衰减dB/km；TX/RX额外光学损耗dB；耦合效率0..1 |
| turbulence | cn2_plane_wave、manual_gamma_gamma或disabled；Cn² m^(-2/3)；α/β；相干时间s |
| pointing | 孔径积分或manual_a0_xi；初始1/e²光束半径m、有效发散半角rad、接收直径m、每轴角抖动rad、固定偏轴rad |
| receiver | 光滤波全带宽Hz；LO功率dBm、线宽/频偏Hz；响应度A/W；每PD暗电流A和单边热噪声密度A/√Hz；PD上限A；电带宽Hz；TIA V/A；ADC位数与满量程V；载波恢复；OSNR参考宽度nm |

默认64GSa/s、4sps→16GBd，16QAM毛标号速率64Gb/s。K7(171,133)卷积码、约1/2码率，再逐编码比特重复2次；净业务吞吐扣除全部终止、填充、前导、包头、CRC、RRC尾部、保护和失败。不是LDPC，不ARQ，不免费反馈。改QPSK可用`qpsk_conv`；改调制/FEC必须记录资源变化，不能视为严格同时间消融。

默认假设1550nm、CW输入8dBm，TX EDFA15dB/NF5dB/总输出上限20dBm，RX EDFA15dB/NF5dB/上限10dBm，1km，接收孔径8cm、固定耦合效率2%。激光输入8dBm**不是实际发射功率**，实际发射功率还受调制占空/电压、调制器/光学损耗及EDFA影响。

good/moderate/stress是三组人为压力配置，不是从实测天气拟合的三个等级：

| case | 衰减dB/km | Cn² m^(-2/3) | 每轴指向抖动μrad |
|---|---:|---:|---:|
| good | 1 | 1e-15 | 10 |
| moderate | 8 | 3e-14 | 30 |
| stress | 15 | 1e-13 | 70 |

case名称不替代实际Rytov值。正式归因应在新配置中每次只改变功率/距离/衰减/Cn²/指向中的一个因素，不把三种损伤合起来解释为“湍流收益”。

## 传播和功率合同

采用复光场包络sqrt(W)，不在64G采样上直接采样约193THz载波。

`sigma_R²=1.23 Cn² (2π/λ)^(7/6) L^(11/6)`，自动模式用声明的平面波、点接收统计近似：
`alpha=1/expm1(0.49*s/(1+1.11*s^(6/5))^(7/6))`，
`beta=1/expm1(0.51*s/(1+0.69*s^(6/5))^(5/6))`，其中`s=sigma_R²`。
超过本版声明的Rytov范围10停止；Cn²=0退化为无湍流。α/β手动模式必须Cn²=null。`I=Gamma(alpha,1/alpha)*Gamma(beta,1/beta)`只在分布上单位均值，不对每帧重归一化。

有效光束半径`w(L)=hypot(w0,L*theta)`，theta为包含衍射的有效1/e²发散半角，不自动模拟湍流展宽，且不得低于λ/(πw0)。每轴高斯指向抖动叠加固定偏移，**数值积分圆孔径上的偏移高斯光强**，同时包含几何收集和指向，不重复乘一个几何损耗。记录的xi是Farid高斯近似的诊断参数，本版孔径积分不使用这个近似；不能拿近似xi当数值积分的精确分布参数。手动A0/xi模式必须其余几何参数全部null。

`P_rx_aperture_signal = P_launch_signal * 10^(-attenuation_db_per_km*L/1000/10) * aperture_pointing_collection * I`。
再乘RX光学透过率和耦合效率，得到耦合后的信号功率。传播光场乘对应总光强增益的平方根和块相位。慢衰落在一张图的probe和所有包间共享；总空口发射时长超过相干时间立即停止。不同图/重复是独立快照，**没有连续跨图湍流过程、空间波前、孔径平均闪烁、自适应光学或单模耦合场积分**。固定耦合效率不能冒充这些物理机制。

## 放大器、噪声和接收机合同

IQ-MZM为两个null偏置子MZM，电场余弦传递，I/Q相位合并；显式50/50分光与合光，选定输出端场系数1/2，另一端承载余能。插损另外计入。未做电极行波/啁啾/实测ER拟合。linear_reference用于小信号对照，不是实际MZM，驱动限制Vpi/2。

EDFA为平坦带宽黑盒，增益G，`N_ASE_per_pol=(F*G-1)*hν/2`（G>1；透明G=1时不产生ASE），模拟带宽Fs内总两偏振ASE功率`2*N*Fs`。同时限制信号＋输入ASE放大＋新ASE的平均总输出；不是泵浦功率模型，也不是“3dB增益压缩点”的饱和曲线。总输出上限仅对**本仿真频率窗口**定义，不冒充整个EDFA增益带的功率。输入已经超过上限/目标则报错，不偷偷把EDFA变成VOA。两个EDFA分别产生独立ASE；TX ASE受路径衰减，RX ASE在接收端新增，不按前面的FSO损耗衰减。

ASE两偏振都计入光功率和PD散粒噪声；只有与LO同偏振的噪声参与LO拍频。理想90°混频器四输出场为`(Es±Elo)/2,(Es±jElo)/2`，四端功率和守恒。各PD平方律探测，I/Q平衡相减，不再额外添加一份重复计算的“ASE等效电噪声”。

每PD单边散粒PSD`2q(I+Id)`，每个实采样方差`q(I+Id)*Fs`；单边热噪声密度i_n对应每PD采样方差`i_n²*Fs/2`，之后电滤波。只用这一热噪声配置，不重复叠加温度/电阻Johnson噪声。PD饱和与TIA/ADC裁剪另记，不当作白噪声。尚未模拟背景太阳光、RIN、TIA非线性、器件响应度频率依赖、ADC时钟抖动或双偏振DSP。

滤波采用明确的假设四阶幅频、零相位传递函数，FFT零填充降低环绕，数值积分采样频段内ENBW。**没有测量硬件响应或真实群时延，空口时间不含器件群时延。** AWG Fs与电带宽不同。ADC与DAC共享已知采样时钟；帧起点由公开256符号前导相关搜索，不读取传播延迟或真实h。

默认线宽/频偏为0、pilot_constant恢复。若开启线宽或频偏，必须选择pilot_cfo_dd：导频相位斜率估频偏，再用公开QAM决策相位环跟踪；允许出现误判/周跳，不用发送数据矫正。不支持实际实验室完整CMA/DD-LMS/Volterra/采样时钟恢复。本振相位在各包及保护间连续推进。

## SNR和输出位置

| 字段 | 定义 |
|---|---|
| power_stages | laser/after_mzm/after_tx_edfa/launch/rx_aperture/rx_coupled/after_rx_edfa/detector_input，各signal/ASE/total W和dBm |
| received_optical_signal_dbm | 光滤波后、90°混频器信号输入端；不含LO |
| received_optical_total_dbm | 同位置信号＋两偏振ASE；不含LO |
| osnr_db | 同位置信号／参考光带宽内两偏振ASE，默认0.1nm；注明滤波和参考Hz |
| analog_noise_only_snr_db | 电滤波后、TIA/ADC前，噪声单独统计的电流功率比 |
| physical_symbol_snr_db | 离线信号/噪声支路经相同匹配滤波、同RX选定样点；干净信号对发送符号的投影能量／随机噪声能量；ADC前等效诊断，不包含ADC误差 |
| estimated_snr_db | 接收导频复增益和残差估计，含ADC/残余ISI等影响；不能当无偏真值，深衰落时有噪声地板偏差 |
| routing_probe_snr_db | 独立probe估计；这是神经门控来源，不是后续各包汇总SNR |
| evm_rms / sndr_from_evm_db | 实际ADC/DSP后载荷符号的误差，包含估计误差、非线性、量化、相位、ISI；-20log10(EVM)不是任意条件下的纯噪声SNR |

SNR汇总取线性能量总和的比值；光功率按活跃波形时长池化W，再转dBm；BER汇总错误比特/总比特。不平均dBm冒充平均功率。噪声为零/信号消失时相关比值输出null并保留能量；不填巨大的有限“真实SNR”。这是模型真值/接收估计，不是硬件实测。

`samples.jsonl`保留每图、每case、每重复的PSNR/SSIM、BLEU2/token_acc、text_CER/WER、恢复文字和仅评价的原文、全量失败、速率、字节、时间和功率。
`packets.jsonl`保留逐包功率/SNR/噪声分解/CRC/前后FEC BER/裁剪；`gallery.html`展示保存范围内所有条件的原图、恢复图、文字和均衡前后星座图；`summary.json`和`REPORT.md`同时提供全业务和成功业务结果。图像128×128处理不变；source_rgb8_bytes是处理后RGB8逻辑大小，不是原JPEG文件字节数。

`scheduled_seconds`为实际TX波形样点＋保护除Fs；`link_seconds_without_mode_feedback`再加一次L/c，未假定ARQ。神经/CPU通信模拟的墙钟时间单独保存。goodput计成功交付语义body，失败对象为0。活跃信号光能量J另记，但没有闲置ASE、泵浦/电路电能或整套设备能效，不声称完整能效闭环。

## 服务器运行（GPU1，不重训）

先按交付命令校验、解压到项目内新暂存目录，再由install.py添加模块。安装器只新增本模块，并为共享HANDOFF.md保存逐字备份后追加新条目；备份/前后SHA位于v10_extension/deployment_records。所有路径均在`/data1/zw/WYH/multimodel`，使用既有Python3.8/Torch1.12，不执行pip升级。

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
export CUDA_VISIBLE_DEVICES=1
"$PYTHON_BIN" -B v10_extension/optical_link_v1/launch.py check
"$PYTHON_BIN" -B v10_extension/optical_link_v1/launch.py link experiment_results/v10_optical_link_diagnostic_01
"$PYTHON_BIN" -B v10_extension/optical_link_v1/launch.py smoke experiment_results/v10_optical_link_smoke_01
```

link只是随机字节，不测PSNR。smoke为2张validation×(清洁量化参考＋3物理case)=8行。原始权重要求：

```text
experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth
experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth
```

如果A之前输出目录不同，传`--a-delta /data1/zw/WYH/multimodel/实际目录/train_A/last_delta.pth`，不要用D或其他基权重替代。运行预算：link15分钟、smoke30分钟、quick2小时是强制上限，不是预测。检查完成及图文展示后再运行：

```bash
"$PYTHON_BIN" -B v10_extension/optical_link_v1/launch.py quick experiment_results/v10_optical_link_quick_01
```

quick默认8张validation、1次信道快照、清洁参考＋3case，共32行。可用`--val-samples 32 --repeats 2 --save-images 8`扩大；这增加评估，不训练新的种子。先根据smoke实测墙钟判断耗时。不自动启动长训练。

```bash
tail -n 20 experiment_results/v10_optical_link_smoke_01.log
"$PYTHON_BIN" -B v10_extension/optical_link_v1/collect.py experiment_results/v10_optical_link_quick_01 v10_optical_link_quick_records.tar.gz
```

给回records.tar.gz；不含权重、COCO或预训练模型。如果失败，给完整日志和run_identity.json，不把失败目录当完成或覆盖重跑。

## 一手依据和非等价声明

- [OptiSystem组件与监测工具](https://optiwave.com/optisystem-overview/)：只参考组件层次和监测组织，没有调用OptiSystem或复制内部算法。
- [OptiCommPy官方器件源码与公式说明](https://opticommpy.readthedocs.io/en/latest/_modules/optic/models/devices.html)：核对ASE每偏振PSD、散粒/热噪声采样单位和相干探测；本版独立NumPy实现，没有引入该库依赖，输出上限黑盒与完整EDFA速率方程不同。
- [Farid & Hranilovic, JLT 2007](https://www.ece.mcmaster.ca/faculty/hranilovic/publications/articles/07/jlt06IEEEFinal.pdf)：采用传播强度分解、Gaussian孔径/指向定义和慢衰落假设；不把该文IM/DD OOK的电SNR公式搬到相干接收机。
- [Al-Habash/Andrews/Phillips, 2001作者机构记录](https://stars.library.ucf.edu/facultybib2000/2907/)：Gamma-Gamma统计模型依据；自动参数为明确披露的平面波近似，不代表完整Gaussian波束/有限孔径实测拟合。

光链路更具体不等于图文恢复必然改善，也不能自动修好已有文字解码器。没有硬件标定或独立模型效能结果，不宣称真实设备等价、模型创新增益或论文投稿成功。
