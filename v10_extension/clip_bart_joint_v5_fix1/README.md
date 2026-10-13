# V5 fix1：只修验证汇总兼容性

状态：服务器日志显示三组smoke训练/评估完成，最后汇总失败。
原因是V4文本controls写四种记录（含matched_beam4），旧V5只接受三种。
不是显存不足；不是联合模型效果通过。修复只改变report.py、test_report.py、
对应MANIFEST.json；不改模型、损失、训练预算、信道、配置或质量阈值。

新校验器要求prefit完整四种记录，全部重新核对文本指标；beam4仍仅辅助展示，
质量门禁仍只看matched/zero/shuffled。联合阶段仍要求原三种记录。
测试包含beam完美但greedy失败、第四类缺失/重复/篡改、源划分错误和未知记录。
本地20报告测试及8核心测试通过；服务器真实GPU再验证尚未执行。

补丁只供已安装原V5的项目。原压缩包、历史权重/结果不删除。
安装前核验356历史依赖、其余V5源码及精确旧版/新版哈希；冲突不覆盖。
写入前备份三目标及HANDOFF到v10_extension/deployment_records/下带时间戳目录，
再原子替换每个目标文件。整体不是跨文件事务；意外中断的混合版本会拒绝重装，
保留备份后定位，不自动回滚、不删除用户文件。相同完整安装可重复验证。
不要再用旧V5解压目录的安装器覆盖fix1。

下载补丁与.sha256到Mac桌面，替换SSH地址上传：

```bash
scp /Users/a1-6/Desktop/v10_clip_bart_joint_v5_fix1.tar.gz /Users/a1-6/Desktop/v10_clip_bart_joint_v5_fix1.tar.gz.sha256 zw@服务器IP:/data1/zw/WYH/multimodel/
```

服务器逐条执行，任何一步出错先停：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_clip_bart_joint_v5_fix1.tar.gz.sha256
test ! -e _v10_clip_bart_joint_v5_fix1 && tar -xzf v10_clip_bart_joint_v5_fix1.tar.gz
"$PYTHON_BIN" -B _v10_clip_bart_joint_v5_fix1/install.py --project-root /data1/zw/WYH/multimodel --dry-run
"$PYTHON_BIN" -B _v10_clip_bart_joint_v5_fix1/install.py --project-root /data1/zw/WYH/multimodel
"$PYTHON_BIN" -B v10_extension/clip_bart_joint_v5/test_report.py
"$PYTHON_BIN" -B v10_extension/clip_bart_joint_v5/launch.py screen experiment_results/v10_clip_bart_joint_v5_02 --max-hours 4
```

固定GPU1，先重新smoke，再正式文本泛化/联合筛查。01失败目录原样保留；
不删除FAILED、不更新原源码快照、不把smoke权重用于正式初始化。
预计smoke5–15分钟、正式筛查30–90分钟仍未实测；4h是上限不是ETA。

```bash
tail -n 60 experiment_results/v10_clip_bart_joint_v5_02/v10_smoke.log
tail -n 60 experiment_results/v10_clip_bart_joint_v5_02/v10_screen.log
```

结束发experiment_results/v10_clip_bart_joint_v5_02/v10_clip_bart_joint_v5_records.tar.gz；
新的质量检查失败也保留结果。不访问test，不升级环境，不杀别人的进程。
工程通过不证明泛化/联合收益。备份中PATCH_PLAN.json记载前后哈希与回滚对象；
需要恢复时先确认无正在运行的本项目任务、当前文件均为记录的after，再逐个
恢复backup三源码与HANDOFF。恢复不会删除本次训练结果。
