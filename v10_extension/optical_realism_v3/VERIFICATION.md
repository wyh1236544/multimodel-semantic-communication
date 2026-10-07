# V3本地验证记录

2026-10-07：本机macOS/Python3.12.14/NumPy/Pillow，无Torch。服务器Python3.8/PyTorch1.12、真实A/COCO/GPU1效果与硬件标定未执行；没有服务器任务由本次代理启动。所有23个本版Python文件按3.8语法解析通过，不等于服务器动态验证。

## 实际执行（退出码均0）

从项目根使用任务运行时完整Python路径运行：

~~~text
python3 -B v10_extension/optical_realism_v3/verify_local.py experiment_results/v10_realism_v3_local_verification_20261007_02
python3 -B v10_extension/optical_realism_v3/link.py experiment_results/v10_realism_v3_local_b2b_20261007_01
python3 -B v10_extension/optical_realism_v3/diagnostic.py experiment_results/v10_realism_v3_local_diagnostic_20261007_02
~~~

测试数量：V3同步7、修复9、核心26；V2同步7、核心24；optical_link_v1核心32、final_compare_v1核心24、physical_link核心11，共140，全部通过。

覆盖实际数值/协议/预算/失败路径：分数定时与ppm/CFO；真值字段改变不影响接收；超捕获范围拒收；16次纯噪声均拒收；固定同波形低SNR回归；完整16点映射与软译码/CRC；未除R的共同误差下限；分半一致性；所有导频/FEC/填充/保护间隔计费；不完整/重复矩阵拒绝；JSON字面后缀即使更新哈希仍被拒绝；未知文件/已有输出拒绝收集。

15个B2B案例（3字母表×5预设）均通过原有宽松功能门限：锁定、EVM²<0.10、数字未编码BER<0.02；EVM约0.166–0.199。不是实验室链路达标证据。

## 同一ADC观测的V2/V3合成回放

24个案例，ideal/combined×3工况×4通道观测，288×16高斯联合特征、相同计费24次重复。每行两个接收器处理同一ADC数组，真值仅供解码后MSE诊断；不是A特征，不是PSNR实验。

| 预设/工况 | V2平均MSE | V3平均MSE | V2→V3锁定数（各4例） |
|---|---:|---:|---|
| ideal/good | 0.000576 | 0.000562 | 4→4 |
| ideal/moderate | 0.987129 | 0.061732 | 0→4 |
| ideal/stress | 0.511066 | 0.272678 | 2→3 |
| combined/good | 0.003536 | 0.003668 | 4→4 |
| combined/moderate | 1.269364 | 0.066038 | 1→4 |
| combined/stress | 0.752873 | 0.276200 | 1→3 |

V3本组无“锁定但MSE>1”，不证明永不错误放行。深衰落仍有拒收，combined/good有小幅误差增加，不是处处改善。小样本合成结果仅支持接收修复方向，不支持真实图文/发表结论。

## 发布检查与未执行事项

构建器生成确定性source-only USTAR及逐文件/依赖SHA清单，在项目内隔离目录核验解压字节、安装/重复安装、HANDOFF按字节备份、冲突拒绝及缺失依赖拒绝；实际结果保存在releases的build_verification.json。构建实际成功后才可交付。

GitHub发布先核对依赖Git blob与当前HEAD，再作带期望HEAD的非强制原子更新；只新增本版源代码并追加共享交接，不携带权重/原始图片/凭据。包SHA、Commit SHA和发布成败以最终交付记录为准。

未执行：服务器Torch模型接口、真实基础+A权重载入、GPU/COCO图文评估、设备实测频响/湍流序列、最终独立test及更多样本。不训练、不加原文旁路、不更改旧结果。先运行README有界pipeline，返回完整记录包，再判断实际恢复收益。

