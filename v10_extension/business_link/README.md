# V3：固定业务量的相干 FSO 通信实验

状态：实现与本地 NumPy 小规模链路验证；服务器 Torch CPU、真实 checkpoint/GPU/COCO 验证尚未执行。不重新训练，不覆盖 V10/V2/已有结果。代码仅新增 `v10_extension/business_link/`。最后更新：2026-10-01。

## 1. 这次改变了什么

主实验固定同一批图片和对应文字，比较不同链路处理实际业务的代价，而不是固定每种方法的压缩后字节数。业务基准是现有模型中心裁剪后的 RGB8 图片字节数，加上数据集规范化第一段描述的完整 UTF-8 字节数；原 COCO JPEG 文件大小另作参考，不能混用这两个压缩比分母。图像 ID、像素/文字哈希和实际字节数冻结于 `business_manifest.json`。

图片和原文仍进入已有联合语义编码器，保持 C32 + 总共256个 I/T 槽、宽度16的模型与权重。文字同时增加独立数字支路：完整 UTF-8 → 分包 → CRC32 → 扰码 → 可选 FEC → QAM → RRC → 相干场信道 → 匹配滤波/前导同步/导频均衡 → LLR/Viterbi → CRC → UTF-8。这里不再用神经文字头的生成句子代替“原文逐字传输”；神经头输出单独保留作为对照。

数字图像支路：归一化4608个语义实数 → 固定范围均匀量化（默认8bit、[-4,4]）→ 位打包与语义对象 CRC → 上述数字链路 → 还原实数 → 原有接收适配器与解码器。它是**数字化语义传输**，不是 JPEG/AVIF 等传统图像源编码基线。连续特征支路保留为 `analog` 对照，不赋予人为 byte BER 或“字节压缩比”。

数字配置统一使用所选码率保护全部特征位，不套用原模拟符号的功率UEP；连续配置仍使用原公开功率表。`analog` 的文字仍走16QAM+卷积码，因此它是“连续特征+数字文字”的混合链路对照，不是纯连续文字传输。能量为归一化功率×秒，没有未经校准的瓦特/焦耳标定。

接收端增加两种算法：`baseline` 保留原解码流程；`received_text` 只使用 CRC 成功的数字文字，在接收端重新经过已有 CLIP 投影和已有文本到图像 refiner，再以固定权重（默认0.25）与原分支融合。新融合没有再训练/对齐，不保证增益；文字失败时完全回退原流程。原文只用于发送端和离线评分，不作接收端未传输提示。

## 2. 光通信设置与边界

- 单偏振相干 I/Q；I、Q 两路各64 GSa/s，4 samples/symbol，故 Rs=16 GBd。这里的64G不是“两通道合计”，也不是64Gbps。
- RRC滚降0.1、跨度32符号，复基带双边占用带宽约17.6GHz，每路 I/Q 最高基带频率约8.8GHz。不等于 AWG/MZM 已具有这样的实测模拟带宽。`--bandwidth` 只是显式假定的零相位低通响应。
- 默认8bit DAC、固定满幅4；记录削顶比例、能量、EVM、导频估计 SNR。前导为公开256符号 QPSK训练序列，数据为所选QAM格式。每个突发独立收发 RRC 尾部和256 AWG样点保护间隔，都计时。
- 16QAM裸线路比特率64Gbps，QPSK为32Gbps；K7 `(171,133)` 八进制卷积码名义码率1/2，1024bit分块、每块6bit零终止、软判决Viterbi。`16qam_repeat2` 在该卷积码之后逐比特重复2次、接收 LLR 相加，名义组合码率1/4。实际有效码率还扣除填充，不能直接拿名义净速率当业务 goodput。
- 512字节固定包体，11字节线上类型/序号/总包数/长度头，4字节CRC；末包零填充、FEC分块填充、QAM补位均进入真实波形。语义对象另外包含原V2量化头和CRC。
- FSO使用现有Gamma-Gamma湍流与指向/路径损耗参数；光强增益 I 对应场幅度 sqrt(I)。每个图片任务的探测、文字、特征和重传共享同一准静态复场，不把重传伪装成独立衰落分集。AWGN是无湍流的相干链路参考。
- SNR输入是单位平均符号功率、16GBd参考的名义 Es/N0；变化采样倍数后按固定噪声谱密度换算。输出SNR为接收导频估计，不是固定等于输入标签，也不是由原图算出来的信噪比。
- 默认距离1000m仅用于传播延迟，**不代表已经由距离计算了真实大气损耗/光功率链路预算**。`--distance-m` 可修改。默认流式发送全部固定业务，传播延迟每批业务只计一次；反馈/重传等待另计。
- 默认固定 reconstruction 模式，是发送/接收双方公开配置，没有免费传递源相关索引。`--mode adaptive` 才启用导频驱动的原三模式、含噪RF请求与经过光链路 CRC/FEC 的模式公告，并计入往返和64 RF符号/1MHz反馈时间。`hybrid/generation` 指原模型配额/神经恢复策略；本入口**不执行扩散生成**。
- 本版先比较显式冗余设置，**未实现自动FEC门限选择**。需先验证速率/可靠性曲线，再在validation上确定门限；不能预先宣称某个未经校准的门限最优。
- 默认不重传。`--max-attempts 2 --ideal-ack-arq` 显式启用有界重传，假设误码-free RF ACK/NACK；每次尝试收取64符号/1MHz确认与2L/c等待，包括最后一次。这不是已实现真实光 ACK 的证据，确认链路自身误码暂未模拟。

未实现/未校准：LDPC、JPEG/AVIF公平基线、激光线宽/相位漂移、载波频偏、时钟漂移、散粒噪声、双偏振、MZM非线性、光功率/接收灵敏度预算、真实大气相干时间、实测器件响应和AWG下载运行。保留这些限制，不能把仿真毫秒/微秒时间写成已实现的硬件吞吐。

## 3. 先解压、校验、安装

上传 `v10_business_link_v3.tar.gz` 及同名 `.sha256` 到 `/data1/zw/WYH/multimodel/`。包不含数据和权重；要求现有 V10 与 V2 的 digital/waveform 已安装且匹配哈希，不升级环境。解压到新的暂存目录，安装器检查全部冲突后才新增目录。

```bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_business_link_v3.tar.gz.sha256
test ! -e _v10_business_link_v3 && tar --keep-old-files -xzf v10_business_link_v3.tar.gz
/home/zw/anaconda3/envs/torch/bin/python -B _v10_business_link_v3/install.py --project-root .
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
export CUDA_VISIBLE_DEVICES=1
"$PYTHON_BIN" -B v10_extension/business_link/launch.py check
```

已存在同字节安装仅校验，不覆盖；旧目录内容不同会停止。若报告 prerequisite 不匹配，请发完整错误，不手动覆盖旧代码。CPU检查包含 NumPy波形与Torch小型适配测试，不下载CLIP/VAE，不做训练。不要停止他人的GPU进程。

## 4. GPU预检与比较

先预检2图、两种信道、SNR20、一个数字配置、两种接收算法，共8次前向，最多30分钟：

```bash
"$PYTHON_BIN" -B -u v10_extension/business_link/launch.py smoke experiment_results/v10_business_v3_smoke_01
```

成功不要求所有衰落样本的CRC/PSNR都达到阈值，而要求输入、完整对照矩阵、有限数值和配对检查通过。查看 `COMPLETE.json`、`business_summary.json` 和图片/文字输出。运行耗时只由该预检推算，不把波形传输时间当训练时间。

建议首次比较8张图+8段文字，FSO中/强、SNR0/10/20，连续/无码16QAM/卷积16QAM/更强冗余16QAM，每项含两种接收算法，共384次前向。最多2小时，**没有额外训练/多训练种子**：

```bash
"$PYTHON_BIN" -B -u v10_extension/business_link/launch.py compare experiment_results/v10_business_v3_compare_01 --samples 8
```

完整默认16张图为768次前向。使用同一业务量增加QPSK对比可添加 `--policies analog 16qam_none 16qam_conv 16qam_repeat2 qpsk_conv`。初期建议先保持单一16QAM数据格式，区分编码收益和调制收益。

按总业务量上限选完整任务（不截断文字/图片，不硬凑压缩后相同字节数）：

```bash
"$PYTHON_BIN" -B -u v10_extension/business_link/launch.py compare experiment_results/v10_business_v3_bytes_01 --samples 100 --business-bytes 500000
```

该命令从冻结validation顺序取尽可能多的完整图文任务，RGB8+UTF8总量不超过500000字节；实际总量写入manifest，通常不恰好等于上限。

复用完全相同业务输入做单独自适应/重传实验：

```bash
"$PYTHON_BIN" -B -u v10_extension/business_link/launch.py compare experiment_results/v10_business_v3_adaptive_01 --reuse-business-manifest experiment_results/v10_business_v3_compare_01/business_manifest.json --mode adaptive --policies 16qam_conv 16qam_repeat2

"$PYTHON_BIN" -B -u v10_extension/business_link/launch.py compare experiment_results/v10_business_v3_arq_01 --reuse-business-manifest experiment_results/v10_business_v3_compare_01/business_manifest.json --policies 16qam_conv --max-attempts 2 --ideal-ack-arq
```

没有自动启动这些附加实验；都用新输出目录，先看主对照。不拿test图片调整融合权重/阈值。本入口仅运行validation。

## 5. 输出与指标

- `business_manifest.json`：图片/文字数量、原始业务基准字节数和逐任务哈希；`design.json`：完整实验矩阵与公开质量阈值。
- `samples.jsonl`：全部条件的PSNR/SSIM、神经输出文字、数字恢复文字、UTF8逐字一致、Unicode字符CER、空格词WER、量化误差、剪裁、模式、比特/样点/时间/能量统计。失败不删。BER和原文对比只用于离线评分，不能修正接收数据。
- `packets.jsonl`：实际每次突发/重试、导频、样点、CRC、pre/post错误比特数和等待。两种接收算法比较同一传输，包记录不重复存两遍。
- `constellations/*.svg` 和 `.json`：先验指定前N张图片的**全部条件**，文字和特征星座；蓝色接收点、红色数字发送字母表。连续特征只有蓝色点。不是挑好看的结果。
- `images/`：同一原图和所有条件恢复图，文件名包含信道/SNR/配置/接收算法。
- `business_summary.json`、`business_table.csv`、`REPORT.md`：无条件PSNR（含失败）、逐字文字一致率、pre/post-FEC BER、PER、EVM、估计SNR、总线路尝试时间、实际仿真计算耗时、吞吐与接收文字融合配对增益。BER为总错误bit/总测试bit，不平均不同长度包的BER。

`gross_modulation_bps` 是裸QAM物理比特率；goodput要明确分母和成功条件。`delivered_business_goodput_bps` 是完整数字特征对象+逐字正确文字对应的**原始业务等效**吞吐；连续支路没有数字对象交付确认，填null。`quality_business_goodput_bps` 要求文字逐字正确并达到预先指定的PSNR（默认20dB，可用 `--task-psnr-threshold` 调整）；含源压缩，可能超过裸线路比特率，这不是违反物理速率。20dB不是普适视觉合格线，PSNR曲线和质量/时间Pareto应同时看。

若任何任务失败，`completion_time_seconds=null`，只报告已花的尝试时间，不谎称“全部数据成功传完”。`COMPLETE.json` 仅表示实验矩阵计算完成。`received_text_fusion_gains` 以图像为单位平均重复后配对bootstrap；这不覆盖多次训练不确定性，也不构成论文正式显著性的充分证据。

## 6. 运行进度和发回结果

程序逐条件/逐图打印行数与耗时。在另外一个终端：

```bash
cd /data1/zw/WYH/multimodel
wc -l experiment_results/v10_business_v3_compare_01/samples.jsonl
pgrep -af 'business_link/(run|launch).py'
nvidia-smi -i 1
```

首次比较预期384行；启动前/加载checkpoint时文件可能还没有生成。不要重复启动相同输出。结束后验证和打包：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/business_link/collect.py experiment_results/v10_business_v3_compare_01 v10_business_v3_records.tar.gz
```

把 `v10_business_v3_records.tar.gz` 发回即可，包含必要报告、逐条记录、图文与星座，不含大权重。若CPU检查或GPU预检失败，发完整终端错误及新目录的 `run_identity.json`，先不扩大运行规模。
