# Fix2 验证与限制

日期2026-10-13，位置本地Mac项目镜像，现有bundled Python，不安装依赖。
已保留Fix1及原V6压缩包、所有历史模块/权重/数据/运行证据。

## 已执行：原错误复现与修复回归

1. 读取生产bart_coco_modal_v2/data.cache：联合digest是图像float32字节+原文，
   teacher是item.text。此前report对纯原文SHA与联合SHA误作相等比较。
2. 真实格式合成fixture改为joint digest与独立text digest明确不同。
   未修核验器运行44测试出现5正向矩阵错误，同用户的teacher报错；1真实
   Torch案例跳过。最小修复后43通过、1真实Torch案例仍跳过。
3. 新5报告案例：真实联合/文本双摘要正向；缺文本摘要拒绝；联合SHA冒充
   文本SHA拒绝；独立BART清单不一致拒绝；teacher短向量拒绝。
   保留原篡改teacher/TRAIN-validation泄漏/SNR/预算/失败记录回归。
4. 新8策略/来源案例执行旧生产digest_input表达式（AST/NumPy tensor接口
   shim，非Torch模型）：联合摘要不改+独立文本；Unicode原文非covered；
   改原文；乱序/重复ID；错划分；错标注文件/ID；错联合摘要；缺BART文本。
   共18core通过，无图像编码器/解码器推理，也不作通信性能证据。

## 发布构建执行的验证

build_release.py成功退出后才交付，真实凭据为项目releases中的
v10_clip_bart_joint_v6_fix2_build_verification.json。构建核对旧备份与已发布
V6/fix1实际归档字节；所有378历史依赖、未改模块及新MANIFEST。
8安装合同覆盖备份/幂等/旧_01和_02失败记录保留、dry-run无写、
用户修改/配置漂移/依赖漂移/混合版本/符号链接/包篡改拒绝。
实际USTAR源包还原、逐文件比对、从还原包安装/重复安装、原V6verify-only、
还原后18core/44report测试、Python3.8 AST及精确HANDOFF备份+追加。
历史V5 core8/report20、V4 core10/report11共49回归另行执行。
GitHub逐blob核验、main expected_sha非force更新及未涉及老文件保留凭据
独立记入github_verification.json；不上传运行数据/模型权重。

## 未执行

本地无Torch/Transformers/预训练/COCO。真实Torch接口测试须服务器通过，
原26Torch/HF合同与saved GPU smoke/正式训练仍必须重跑，不保证之后无错误。
旧_02不修改，不补写缺失文本摘要，不删除FAILED重新封成功；新_03运行。
不能从本地合成矩阵推出训练收益、图文联合优势、FSO抗噪或硬件符合性。
