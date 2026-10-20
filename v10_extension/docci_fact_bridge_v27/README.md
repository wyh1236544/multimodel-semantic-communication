# V27 第一阶段：先确认接收图像能补哪些事实（不训练）

2026-10-20。只用现有 V26 完成记录和 V25 已生成 caption，**不重跑 BLIP/BART、不加载权重、不占 GPU、不下载、不安装依赖**。Python 3.8 标准库即可。旧代码、数据、权重、记录和原 WER 门槛不改。

这是已讨论方案的第一阶段，不是已经实现/训练好的新纠错模型。人工事实审核尚未完成，因此本版有意没有 train 命令，也不自动批准训练。[固定协议](PROTOCOL.md)，[后续训练设计与未实现边界](FUTURE_TRAINING.md)。

## 1. 上传源码并校验安装

将 tar.gz 和 .sha256 上传到服务器现有 multimodel 目录，不上传任何新模型权重。

```bash
cd /data1/zw/WYH/multimodel &&
sha256sum -c v10_docci_fact_bridge_v27.tar.gz.sha256 &&
test ! -e v10_extension/docci_fact_bridge_v27 &&
tar -xzf v10_docci_fact_bridge_v27.tar.gz --keep-old-files &&
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_fact_bridge_v27/test_core.py
```

本版本只接受此前已回传核验的 `v26_rx_diagnose_01` 原始字节，错版本/改动过的父记录会停止。不要手改旧 JSON 来匹配。

## 2. 准备现有记录：只有 CPU 文件校验，没有训练

```bash
cd /data1/zw/WYH/multimodel &&
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_fact_bridge_v27/run.py prepare \
  experiment_results/v27_fact_audit_01 \
  --v26-root experiment_results/v26_rx_diagnose_01 --max-minutes 10
```

成功末行是 `V27 PREPARED 32 references / 160 existing outputs; facts PENDING. No training.`。
10分钟是保护上限，不是预计耗时；这一步仅校验数十MB文件，实际耗时应以运行记录为准。
输出 `references.html` 是第一步填写页，`index.html` 为导航。将这些 HTML 下载到本地浏览器打开即可，数据内嵌，无联网需求。

如果本地已有 Codex 给出的同一份准备页，可直接用它审核，不必再在服务器重复准备。
源码也支持已安全解压的原始 records 文件夹（不是 tar 文件本身）：

```bash
cd /data1/zw/WYH/multimodel &&
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_fact_bridge_v27/run.py prepare \
  experiment_results/v27_from_records_01 \
  --v26-records experiment_results/your_safe_review/v26_records
```

## 3. 第一步：只根据原文列事实，先冻结

在 `references.html` 填审核者代号，记录此前是否看过这些候选，勾选人工核对声明。
全部32条原文各选3–30条原子事实，包含事实内容、类别及从原文复制的连续证据。例如把“红色的车旁有两棵树”拆成“有一辆车”“车是红色”“有两棵树”，而不是把整段当一条。

类别：objects物体、attributes属性、colors颜色、counts数量、relations关系、actions动作、other其他。
visibility：potentially_visual可能能看图核实、not_visual非视觉事实、uncertain不确定；本包没有图片，不能把“可能可见”当成已经核实。

网页不会自动保存或联网。随时点“下载当前填写的JSON”保存进度；下次导入该文件继续。最低3条只是最小覆盖要求，不能代表已覆盖全部原文事实。
第一步看不到候选，但此前已看过V25/V26者仍应如实填 exposure，不宣称严格盲法。

下载得到 `V27_references_human.json` 后上传到服务器已有 multimodel 目录，再执行：

```bash
cd /data1/zw/WYH/multimodel &&
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_fact_bridge_v27/run.py seal-facts \
  experiment_results/v27_fact_audit_01 V27_references_human.json \
  experiment_results/v27_fact_review_01
```

会生成新的 `rate_candidates.html`。原准备目录和原事实提交不覆盖。

## 4. 第二步：审核160份候选，方法名不在网页中

下载 `rate_candidates.html` 到本地打开。每条冻结事实选择 retained保留、contradicted明确矛盾、missing未提及、uncertain无法判断。
保留/矛盾要复制候选中的连续证据；missing证据留空；uncertain必须解释。额外记录原文未支持的新增事实，没有则留空列表。

“原文未支持”不等于“图片中为假”。风格仍可能暴露方法，不保证完全盲法。事实原子性/语义判定由审核者负责，程序只核对绑定、覆盖、证据与算术，不能认证审核者身份。

保存 `V27_ratings_human.json` 后上传，执行：

```bash
cd /data1/zw/WYH/multimodel &&
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_fact_bridge_v27/run.py score \
  experiment_results/v27_fact_review_01 V27_ratings_human.json \
  experiment_results/v27_fact_scored_01
```

遗漏任何样本/事实、错引用证据或空表不会静默得分。有 uncertain 会给上下界并保持 `PENDING_ADJUDICATION`。
全部完成也只表示已提交人工审核，不自动宣布互促成功或开启训练。

## 5. 独立复核和回传

任何阶段都可独立核验：

```bash
cd /data1/zw/WYH/multimodel &&
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_fact_bridge_v27/run.py verify \
  experiment_results/v27_fact_scored_01
```

完成评分后打包：

```bash
cd /data1/zw/WYH/multimodel &&
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_fact_bridge_v27/collect.py \
  experiment_results/v27_fact_scored_01 \
  experiment_results/v27_fact_scored_01_records.tar.gz
```

将该 PRIVATE records 发回即可。若只准备完成，也可把上面两个 scored 名称换为 audit，收集准备阶段；这仍不包含任何事实评分。
也可直接把两份已填写 JSON 发给本会话，用同一源码在本地继续处理，省去服务器往返。

所有网页、原文、候选、人工标注、records均 PRIVATE，不可公开到GitHub。源码/协议及真实聚合状态才可同步。
失败保留现场，新尝试用新目录/新包名；不自动重试、不改已有 Conda 环境或别人的任务。
