# V10 Common 分支审计（只新增，不重新训练）

沿用已完成的 V10 pilot 最佳权重。固定 validation 前100图、seed42、每图3次信道重复、AWGN / Rayleigh / FSO moderate、SNR -5/0/10 dB。真实 Common-only 与随机 Common-only 两组共5400次前向；两组各最多1小时（退出清理可能另需约1分钟）。不启动扩散模型，不修改原权重或既有评估。使用 --no-amp；这不是对 AMP 非有限梯度问题的根治。

从 multimodel 根目录运行：

```bash
python -B v10_extension/common_audit/audit.py plan
CUDA_VISIBLE_DEVICES=1 python -B -u v10_extension/common_audit/audit.py run
```

默认权重：experiment_results/v10_pilot_noamp_seed42/best.pth。
默认输出：experiment_results/v10_common_audit_seed42，必须是不存在的新目录。
可先加 `--samples 4 --output experiment_results/v10_common_audit_smoke` 做小样本链路检查；4图检索不能当正式效果。正式实验不从 smoke 续跑。

如两组评估完整，但报告阶段失败，修复原因后使用 `report` 动作；它不会重跑模型。已经存在 REPORT / audit_summary 时不要手工覆盖，先保留证据再排查。

报告读取已有 v10_zero_shot_val100_noamp 和 v10_pilot_val100_noamp 的完整评估，展示绝对 PSNR/SSIM/BLEU2。缺失时明确标记，不伪造结果，也不自动重跑。

输出 REPORT.md、audit_summary.json、captions.html、return_results.zip。HTML 固定展示保存的前4张图在各条件下的第0次重复，不按效果选图。下载整个 return_results.zip，解压后打开 captions.html；将 zip 发回进行分析。原始权重和数据集不在包内。

## 判断标准

1. 真实 Common 的检索 R@1、语义余弦、短描述质量是否优于随机 Common，尤其关注低 SNR 和 FSO；不能用模糊图色调替代语义证据。
2. 检查真实−随机的按图聚类95%区间、模式头拒绝和传输量。原评估保存检索汇总，不保存逐图检索命中，因此不伪造 R@1 置信区间。
3. 随机高斯 Common 是有用信息对照，不能单独证明严格解耦或图像特异信息；下一步可加错配真实 Common、独立对象指标。
4. 这里仍是开发验证集，包含选模图。不能当独立 test；不能把多个条件的逐项区间当整体显著性保证。Common 与 full 传输量不同，full 仅提供背景。
5. 如果 Common 只在余弦上改善、检索/短描述无改善，暂不追加大训练，先检查语义头、截断和指标偏置；若多指标支持作用，再做关闭 Common 辅助任务重训等因果消融。

## 工程边界

遵守 ../docs/engineering-habits.md。所有运行、缓存、输出留在 multimodel 内。启动前核对基线源码哈希；发现差异停止，不覆盖用户文件。安装只新增 common_audit 目录，同名内容不一致时在写入前停止。既有代码、权重、数据、实验不删除。

本版本仅编排既有评估器并生成报告；本地语法/合成测试不代表真实 GPU 性能测试。真实服务器评估须由用户运行。
