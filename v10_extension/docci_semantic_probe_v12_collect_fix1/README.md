# V12 independent collection fix1

修复状态：针对 `report.render()` 使用 `json.dumps` 却没有导入 `json` 的工程错误。
它同时影响文本、图像结果页与旧report命令行，不是模型训练错误。
本补丁只修复collect入口，不修改旧report命令行。

旧V12训练、评价和源码快照有严格SHA关联，因此**不要直接修改旧report.py、MANIFEST.json或结果source_snapshot**。
新增CPU入口在一次调用中给原始report模块提供标准库json，调用原始collect与verify，最后恢复模块状态。
所有验证、质量门槛、source hash、原图/文本、训练权重都不改变。
新结果包含补丁源码和COLLECTOR_FIX1.json，保留生成依据。

## 服务器操作

上传代码压缩包和sha256文件到已有multimodel根目录，然后：

```bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_docci_semantic_probe_v12_collect_fix1.tar.gz.sha256
test ! -e v10_extension/docci_semantic_probe_v12_collect_fix1 && \
  tar -xzf v10_docci_semantic_probe_v12_collect_fix1.tar.gz --keep-old-files
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_semantic_probe_v12_collect_fix1/test_collect.py
```

补丁已解压且测试通过时不重复解压。目录内容冲突则保留现场，不覆盖。
之前失败会留下`.tar.gz.contents`，原样保留，改用collect02名称：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_semantic_probe_v12_collect_fix1/collect.py \
  experiment_results/v10_docci_semantic_probe_v12_01 \
  experiment_results/v10_docci_semantic_probe_v12_01_records_collect02.tar.gz \
  --max-minutes 30
```

这只重新做CPU验收、复制与打包，不用GPU，不运行launch/train，不续训。
30分钟为停止上限，不是预计耗时。看到`Send ...`与`SHA256 ...`后，将collect02压缩包发回。
若collect02也已有文件或contents，保留原物、改collect03，不能删除以绕过保护。

## 验证边界与交接

新增测试覆盖含text/image的真实render分支、main/smoke、HTML转义、原始收集器打包、失败传播、已有目标保护和源结果不变。
输入为明确标记的合成CPU记录，不代表V12真实模型质量；旧V12 CPU合同测试仍须单独通过。
本地没有本轮服务器记录，不能声称收集已完成或训练效果变好。
原始16项CPU合同、526个历史依赖和V12源文件仍按旧MANIFEST核验。
没有新增依赖、pickle/Torch读取或服务器任务，没有变更FSO、参数、训练步数或质量门槛。
