# 最终方向对照 V1（冻结A，不再训练）

这一步直接回答：在完整图文业务、相同光链路和公开资源上限下，值得继续的是传统数字、语义数字，还是语义连续传输？代码完成不等于实验已经完成，也不等于达到论文投稿要求。

## 三条路线与三项定位对照

| 路线 | 发射端 | 接收端 | 能回答的问题 |
|---|---|---|---|
| traditional_digital | 同一128×128图像→JPEG；完整原文→UTF8/zlib；独立CRC对象→分包/FEC/16QAM | 解调/FEC/独立CRC→JPEG图像与原文 | 传统有损图像+无损文字基线有多强？ |
| semantic_digital | 冻结A→C32/I224/T32，每token16维→8 bit量化→CRC/FEC/16QAM | 语义对象CRC→冻结A的图文解码器 | 语义表示在数字光链路是否有资源−质量收益？ |
| semantic_analog | **同一份**归一化4608实数→两实数一复符号→公开预算内重复→I/Q光调制 | 导频均衡、重复合并、LMMSE→同一A解码器 | 冻结A连续接口的兼容性及退化是否更平滑？ |

另外保存JPEG无信道、语义浮点无信道、语义Q8无信道。由此区分表示/解码、量化与光链路各自造成的损失。

语义分支不含原文旁路；传统分支的原文也**不**注入语义分支图片解码器。本轮不重新训练文字辅助图像模型，不重新启用自适应模式，完整保留历史模块与结果。

重要边界：A权重适配过量化数字接口。连续分支没有专门重训，不能称为优化后的DeepJSCC，更不能由其失败推出所有连续语义通信无用。图文同时完全还原与保留语义也是不同目标。

## 默认物理链路及公平性

沿用`optical_link_v1/optical_config.json`：激光→I/Q MZM→TX EDFA→发射光学损耗→自由空间/Gamma-Gamma/指向/孔径→RX EDFA→相干LO/平衡探测→TIA/ADC→DSP。

- AWG 64 GSa/s，4 SPS，16 GBd，RRC滚降0.1、跨度32符号；复基带占用带宽17.6 GHz。
- 数字默认16QAM，毛速率64 Gb/s；扰码、K7卷积码、每1024源bit独立终止、编码bit重复2、512 B定长包、包头/CRC/填充；有效码率并不是无开销的1/2。
- 每个业务时隙上限4 µs，即256000 AWG样本。计入独立探测导频、每包256训练符号、RRC尾巴、保护间隔、FEC/重复/填充。传播时延另加一次L/c，不将GPU执行时长当光链路时长。
- JPEG从quality 1..95枚举，选择**最高可装入完整线速预算的quality**，固定4:2:0、不启用progressive/optimize；不使用恢复PSNR或信道真值挑quality。JPEG实现选项依据[Pillow官方文档](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg)。记录Pillow版本和实际编码字节。
- 默认语义数字约3.460 µs。连续2304源复符号能重复27次，实际约3.932 µs；所有重复计费、同一慢衰落场，**不是27次免费独立分集**。
- 三种方法每个活动burst的**发射信号光功率**相同，拥有同一时间/信号能量上限；实际消耗可不同。信号能量不含空闲ASE或仪器电功耗。
- 发射端仅利用已知MZM/EDFA模型调整激光功率，使目标launch signal power成立；不能满足激光/EDFA上限就报错，不悄悄截断。记录实际激光功率、每级signal/ASE/total power及接收功率。
- good、stress各一个−9 dBm信号发射点；moderate扫−24、−21、−18、−15、−12、−9、−6 dBm，共9工况。**扫发射功率，不预设接收SNR**。详见`plan.json`。
- 相同图像/文字/浮点payload、相同衰落和独立导频观测。不同长度波形的数据噪声数组不声称逐点完全相同。离线物理SNR不送入解码门控，解码器仅得到导频估计（限幅−40..40 dB）。
- 此版锁定pilot_constant、零CFO/线宽。参数为假设值，未硬件标定；不是完整OptiSystem替代。

## 上传、校验与添加安装

先将`v10_final_compare_v1.tar.gz`和对应`.sha256`上传到已有服务器项目，不直接覆盖旧模块。

```bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_final_compare_v1.tar.gz.sha256
FINAL_STAGE=$(mktemp -d -p . final_compare_unpack_XXXXXX)
tar -xzf v10_final_compare_v1.tar.gz -C "$FINAL_STAGE"
/home/zw/anaconda3/envs/torch/bin/python -B "$FINAL_STAGE/_v10_final_compare_v1/install.py" --project-root /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B "$FINAL_STAGE/_v10_final_compare_v1/install.py" --project-root /data1/zw/WYH/multimodel --verify-only
```

安装先校验全部旧依赖和新增源文件SHA256；已有同名模块只允许逐字节相同，不覆盖修改。共享HANDOFF先逐字节备份再追加。解压不含macOS扩展头，不包含数据/权重/结果。

## 一条命令跑到完整结果

默认路径：V10 mixed seed42 base=`experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth`；A delta=`experiment_results/v10_token_guidance_quick_01/train_A/last_delta.pth`；COCO、CLIP、VAE与split沿用原路径。严格核对base/delta/split SHA和delta参数键。

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/final_compare_v1/launch.py pipeline experiment_results/v10_final_compare_01 --max-hours 4
```

自动顺序：CPU契约测试→2图×3工况×1信道观测smoke→独立核验→32图×9工况×3观测validation→独立核验→图文/星座/指标汇总→结果压缩包。3次是信道随机观测，不是3次训练或新增权重。主评估含2688行（2592传输、96无信道对照）。示例固定保存前8张图全部工况/观测，不按效果挑图。

始终指定GPU1；程序内cuda:0是隔离后的物理GPU1。不得停止其他人的GPU任务。当前环境可用显存不足会停止并保留日志，不安装或升级依赖。整体4小时上限只控制本流水线创建的子进程。无COMPLETE的部分结果不能当完整实验。

需要退出终端后继续，可改用以下命令；输出名与日志必须是新的：

```bash
test ! -e experiment_results/v10_final_compare_01 && test ! -e experiment_results/v10_final_compare_01.launcher.log && nohup /home/zw/anaconda3/envs/torch/bin/python -B -u v10_extension/final_compare_v1/launch.py pipeline experiment_results/v10_final_compare_01 --max-hours 4 > experiment_results/v10_final_compare_01.launcher.log 2>&1 &
```

仅检查/仅smoke/仅主评估也支持`launch.py check`、`launch.py smoke 新目录 --max-hours 0.5`、`launch.py validation 新目录 --max-hours 4`。需要自定义输入路径时直接用`run.py --help`；不改冻结权重或manifest。保存图片数必须不超过样本数。

## 查看进度和取回结果

```bash
tail -n 12 experiment_results/v10_final_compare_01/validation.log
cat experiment_results/v10_final_compare_01/validation/progress.json
find experiment_results/v10_final_compare_01 -maxdepth 2 -name COMPLETE.json -print
pgrep -af 'final_compare_v1/(launch|run).py'
nvidia-smi -i 1
```

smoke阶段先查看`smoke.log`；只有主评估开始后才有validation.log。progress每处理完一张图更新，包含已经完成/预计总行数、耗时和粗略总时长预测。可以先估1–3小时；不同CPU/FEC负载和长连续波形会改变耗时，以本次记录为准，不能保证一定在上限内完成。达到上限保留结果，不伪造完成标记。

最终给我：`experiment_results/v10_final_compare_01/v10_final_compare_records.tar.gz`和SHA256侧文件。它包含报告、全部真实原文/恢复文字、图片、星座、物理功率/SNR、逐包误码与完整身份；不含权重。

本地解压后双击`validation/report.html`可离线筛选工况/方法/图片编号；阅读`REPORT.md`及`summary.json`判断收益。失败样本均进入全样本统计。默认“PSNR≥20 dB且归一化WER≤0.1”只是公开业务代理阈值，原文完全一致性另报；聚类bootstrap以图像为独立单位，不能把96观测当96独立图像。

## 何时算本轮完成

smoke通过、2688行严格匹配、所有资源与逐包计数核对、真实恢复图文及报告生成、COMPLETE通过校验、结果打包成功。能够据此选主研发路线并定位文字瓶颈，但不能自动宣布投稿成功或所有连续通信无效。方向/参数锁定之后才考虑新的独立test验证；本流水线**不会**访问test做选择。
