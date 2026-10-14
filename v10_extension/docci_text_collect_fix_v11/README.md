# V11 仅收集修复：FP32诊断汇总

状态：CPU数值/合同测试已执行；这轮服务器完整原始记录尚未发回，全部31项需在服务器重新校验。
原训练和100条评价已结束，不要重启pipeline或重训。此包不加载Torch、GPU、BART、权重pickle。

原诊断直接在FP32张量求均值，而旧校验将分段FP32均值转成Python数值后加权，固定rel_tol1e-7会拒绝合法舍入差。用户打印的五项差值为5.83e-8..3.32e-7；这是舍入量级证据，不是整轮数据已通过。

## 修改范围

只对五种POST-HOC诊断均值的加权汇总采用计数相关的FP32误差界。所有原始数值保留，不把重算均值写回；原WER、EOS、margin质量门槛不改。原代码/MANIFEST/源码快照、训练/评价/源信息/字节哈希/波形矩阵/保存检查点核验不改。
新进程临时替换旧diagnostic函数，其余校验全用旧实现；非汇总部分经AST对比完全一致，退出/异常恢复原函数。超过误差界、非有限数、非FP32均值、错误数量、错误帧、源泄漏等仍拒绝。

完整校验原结果一次，再把轻量文件按SHA复制到新收集目录，生成旧定义summary和全部100篇HTML。原失败目录只读，权重/RNG/优化器/缓存留服务器；不是重新训练或重新评价。新的verification_fix/receipt.json保存每一项误差/误差界、旧拒绝情况、原路径和逐文件身份；包内含新校验器源码快照。

## 上传、解压、安装

源码包与sha256文件上传到`/data1/zw/WYH/multimodel`：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
sha256sum -c v10_docci_text_collect_fix_v11.tar.gz.sha256 &&
test ! -e _v10_docci_text_collect_fix_v11 &&
tar -xzf v10_docci_text_collect_fix_v11.tar.gz --keep-old-files &&
"$PYTHON_BIN" -B _v10_docci_text_collect_fix_v11/install.py --project-root .
```

先检查517个历史源码依赖与新文件冲突，新增模块；准确备份旧HANDOFF后追加，不覆盖旧模块。若冲突、旧代码被改或文件缺失，保留并发回错误，不删除/跳过检查。

## 只重新校验和收集

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_text_collect_fix_v11/test_fix.py
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_text_collect_fix_v11/collect.py \
  experiment_results/v10_docci_text_continue_v11_20261014_201245 \
  experiment_results/v10_docci_text_continue_v11_20261014_201245_recollect02
```

不用GPU，无训练，不需要nohup整轮pipeline。不再有180秒硬限制；30分钟是CPU安全上限而不是预计时间。剩下是磁盘读取/哈希、校验、HTML和压缩，实际耗时未测。新目录存在时保留，改成recollect03；原目录无需删除或改标记。

成功显示`Send ...`。发回：

```text
/data1/zw/WYH/multimodel/experiment_results/v10_docci_text_continue_v11_20261014_201245_recollect02/v10_docci_text_continue_v11_records_collect02.tar.gz
```

该目录中的`actual_text_recovery.html`下载后用浏览器打开；`summary.json`看WER/截断/重复和旧质量状态。`RECOVERY_COMPLETE.json`表示新收集已完成，旧QUALITY_STOP仍可以存在，不能把工程修复当性能改善。

## 数值规则与验证边界

令u32=2^-24，gamma(k)=k*u/(1-k*u)。对于相同非负FP32 token值的整体和分段均值，使用整体gamma(2N+4)、最大分段gamma(2M+4)，另计双精度加权和舍入；由加权均值反推保守正值误差界，绝对下限沿用1e-8。N<=510/M<=62/非空段<=9；只适用于该发布的FP32诊断，不扩展到WER/资源/哈希或训练门槛。

PyTorch1.12.1关于单精度、运算次序、整体/切片计算差异的原始说明：
https://raw.githubusercontent.com/pytorch/pytorch/v1.12.1/docs/source/notes/numerical_accuracy.rst

16本地CPU测试包含五个打印数值的回归（非全部31原始记录）、500组合成非负FP32规约、真实差值拒绝、NaN/负值/非FP32/容量边界、原始源/帧/数量/哈希/诊断标记负向检查、非汇总AST一致性、只读复制/排除权重及失败恢复。旧V11的8core+11report也回归。源码恢复/安装冲突检查另存构建记录；没有训练速度、模型效果或服务器全量通过的本地声明。
