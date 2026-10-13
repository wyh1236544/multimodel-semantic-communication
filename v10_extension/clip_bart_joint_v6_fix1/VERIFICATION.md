# V6 fix1 验证与限制

核验日期2026-10-13；运行位置本地Mac项目镜像，不是学校服务器。
解释器是现有Codex bundled Python；未安装或升级任何依赖。

## 根因复现与修复回归（已执行）

- 输入用户GPU日志：三组smoke训练/两条件评估完成，finish的report核验拒绝SNR。
- 生产 clip_bart_joint_v5.data.clean_args 不改，有限clean条件占位为40dB；
  V6 evaluate读取真实args[2]保存NPZ，而旧report固定要求20dB。
- 新合成fixture改为执行生产clean_args函数BODY（AST提取、NumPy接口shim）；
  这不是Torch/GPU运行，明确与单独真实Tensor测试区分。
- 未修report上39项运行出现4项正向矩阵错误（同报Received pilot SNR differs），
  1项实际Torch测试跳过。修正report后38通过，1项实际Torch跳过。
- 新7测试：production40roundtrip；actualTorch40roundtrip；clean20拒绝；
  clean0拒绝；physical浮点32舍入允许；physical错配拒绝；NaN/±Inf拒绝。
- 原32报告用例保留，包括full clean/paired-FSO合成正向矩阵、泄漏/篡改/
  丢失/失败状态拒绝。物理容差保持1e-4，质量门槛、模型和发送资源不变。
- V6策略10项通过；历史V5 core8/report20、V4 core10/report11，共49项通过。

## 发布构建检查

build_release.py必须成功退出才交付；其实际凭据保存在项目releases中
v10_clip_bart_joint_v6_fix1_build_verification.json，而不是把本文件当完成标记。
构建执行8项安装合同：字节备份/旧结果保留/幂等、dry-run无写入、用户源码
冲突、配置漂移、历史依赖漂移、混合版本、符号链接、修复包篡改拒绝。
归档为固定USTAR/source-only；实际还原并逐文件比对，安装到原V6镜像，
重复安装、原V6 verify-only以及还原后的报告测试；检查Python3.8 AST，
378历史源码SHA和精确HANDOFF备份+追加。旧V6 tar保持原字节。

GitHub同步基于已核实的main头，expected_sha租约/非force更新。逐文件Git blob
核对及所有未涉及的老文件保持证据另存github_verification.json；本地测试
不是GitHub保存凭据。只保存源代码/文档/manifest，不上传权重、数据和结果。

## 未执行事项

本地无Torch/Transformers/预训练/COCO，真实Torch接口回归跳过；服务器launcher
check仍须执行它及原26 Torch/HF合同。修复后的实际saved GPU smoke/正式
训练/扩散未执行，无模型增益或完整运行成功结论。保持失败_01不变，新_02
从同正式V5文本父权重重新预检。禁止删除失败标志或手改历史NPZ以重新封成功。
