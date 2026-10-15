# V13：离线语义评价，不是新的训练模型

状态：2026-10-15 新增。历史模型/划分/权重/解码设置/旧 WER 与 QUALITY_STOP 不变。
V11/V12 的结果已看过，因此这是**事后探索性补充评价**，不是未看数据的预注册实验。
`protocol.json` 在计算新语义分数前冻结；之后不得根据成绩换评价器、抽样或事实规则。
公开源码/协议/合成测试，不公开原始 DOCCI 图文、完整恢复文本或权重。

## 本版做什么、不做什么

- 只读已保存的 V11/V12 JSONL 或 records tar；不解压、不执行 records 内代码、不读取 checkpoint。
- 同一 qual_dev 100 条：V11、V12 CE、V12 CE+KD 各 matched/shuffled/zero；另含原生 BART 和固定错配原文，共 **1100 个文本输出**。
- 从字符串独立重算历史规范化 WER，检查完整矩阵、原文哈希、划分、截断和跨版本源文本一致性。
- 以原文事实保留率、原文不支持的新增事实率为主；抽取 **30 条描述、全部 11 组，共 330 个候选输出**审核，不是只看 30 个候选。
- 自动事实评分 **未实现**：不把关键词命中率冒充 SPICE，也不让 LLM 自己打分。人工审核模板为空就明确 PENDING，不生成假分数。
- BERTScore 为可选辅助；固定独立 Longformer 而非 CLIP/BART。无服务器依赖变更、无自动训练、无 FSO/数字仿真、无自动“质量通过”。

## 为什么不是默认 RoBERTa 重缩放

DOCCI 原文虽不一定超长，清零/错配接收后的重复输出也必须完整计入。固定评价器为
`allenai/longformer-base-4096`、layer7、slow tokenizer、IDF=false、CPU、bert-score0.3.13。
Longformer 支持长上下文；layer7 来自该版本 BERTScore 的既有模型设置，不是在本项目上挑分数。
**仍要用实际 tokenizer 审计所有完整文本，超过 4096（含特殊 token）整次停止，不截断、不删除失败样本。**
本包未提供可核验的该模型官方重缩放表，因此诚实报告 raw P/R/F1 与错配描述参照，不编造 baseline。
官方 rescale 是线性变换，不能解决事实识别问题，也不是“准确率百分比”。若将来启用重缩放或更换评价器，必须新协议并同时保留本协议结果，不挑最高分。

## 1. 服务器只需部署、测试（无需安装新依赖）

先把 `v10_docci_semantic_eval_v13.tar.gz` 和 `.sha256` 上传到 `/data1/zw/WYH/multimodel/`：

```bash
cd /data1/zw/WYH/multimodel
sha256sum -c v10_docci_semantic_eval_v13.tar.gz.sha256
test ! -e v10_extension/docci_semantic_eval_v13 && tar -xzf v10_docci_semantic_eval_v13.tar.gz --keep-old-files
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_semantic_eval_v13/test_eval.py
```

源码包顶层就是 `v10_extension/docci_semantic_eval_v13/`，**无 install.py**，不要重复旧版本的暂存安装流程。
同名目录存在时停止并核验，不删除、不覆盖。归档只有新增模块，不改服务器旧共享文档或旧运行目录。
CPU 合同通过不代表真实 BERTScore 已运行。

## 2. 本地准备（推荐；你已有两个 records 包）

在 Mac 解压代码包到现有项目根目录后运行；也可把同样命令放在服务器，用两个 records 的实际完整路径。
以下本地路径对应当前交接工作区，不包含凭据：

```bash
cd /Users/a1-6/.codex/.chatgpt-projects/g-p-6ab14f6782ec8191b8378af1a045f263/work/multimodel_v10_experiment_closure
/Users/a1-6/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B v10_extension/docci_semantic_eval_v13/run.py prepare \
  --v11-records /Users/a1-6/Desktop/v10_docci_text_continue_v11_records_collect02.tar.gz \
  --v12-records /Users/a1-6/Desktop/v10_docci_semantic_probe_v12_01_records_collect02.tar.gz \
  --output experiment_results/v13_semantic_local_attempt02
```

若输出已存在用 attempt03，不覆盖。`actual_text.html` 展示全部原文、恢复和对照；
`lexical_summary.json` 是重算字面指标；`PREPARED.json` 标明语义指标尚未测量。
`PREPARED.json` 记录输入文件、消耗 JSONL、协议和评价源码 SHA-256；不是对整个历史训练包重新验收。
原始输入不会修改。备份/完整训练核验仍以 V11/V12 的原始验收材料为准。

## 3. 人工事实审核：先原文，再候选

1. **不要编辑准备目录里的模板**。复制 `source_facts_template.json` 到新的工作文件 `source_facts_reviewed.json`。
2. 指定 `inventory_reviewer`。只看原文，逐条填写全部可核查原子事实，例：

```json
{"fact_id":"F01","category":"count","proposition":"Two dogs are present","reference_quote":"two dogs"}
```

`reference_quote` 必须是原文中真实连续片段。分别写对象/数量/属性/动作/关系/否定，不把一长句只当一个事实，不事后只选模型能恢复的事实。每个事实等权。每条完成后设 `inventory_complete:true`。

```bash
/Users/a1-6/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B v10_extension/docci_semantic_eval_v13/audit.py make \
  --prepared experiment_results/v13_semantic_local_attempt02 \
  --facts experiment_results/v13_semantic_local_attempt02/source_facts_reviewed.json \
  --output experiment_results/v13_fact_audit_attempt02
```

3. 将 `candidate_judgements_template.jsonl` **复制为新文件** `candidate_judgements_reviewed.jsonl`；
   给审核者看候选模板与原文，但**不要提供 PRIVATE_model_key.jsonl、模型汇总和原始准备目录**。
   这是流程性身份遮蔽，不承诺审核者绝对无法猜组别。最好由未看成绩的人审核，并抽查分歧。
4. 原文事实状态：`entailed / contradicted / not_mentioned / unclear`。
   其中正确或冲突判断要附候选中的 `candidate_quote`，不能仅以相同关键词判关系正确。
5. 另在 `prediction_facts` 列出候选中全部**去重的原子断言**：

```json
{"fact_id":"P01","proposition":"A blue car is present","candidate_quote":"blue car","state":"unsupported","reference_quote":"","note":"Not supported by source text"}
```

   状态 `supported / contradicted / unsupported / unclear`；有原文支持或冲突时补原文证据。
   原文没说不等于照片里一定不存在，所以指标名称是“原文不支持”，不是视觉真伪的绝对结论。
   空输出的新增事实率记 null，而不是完美零；遗漏率/保留率仍计入。重复断言去重，另有重复指标。
6. 填写 `reviewer` 和 `prediction_inventory_complete:true`。未审核/不完整/缺对照会停止，不给出选择性成绩。

```bash
/Users/a1-6/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B v10_extension/docci_semantic_eval_v13/audit.py score \
  --prepared experiment_results/v13_semantic_local_attempt02 \
  --audit experiment_results/v13_fact_audit_attempt02 \
  --judgements experiment_results/v13_fact_audit_attempt02/candidate_judgements_reviewed.jsonl \
  --output experiment_results/v13_fact_scores_attempt02
```

主指标先按每条描述平均，再按组平均，同时报告微平均计数、含糊判断与空输出数。
原文事实保留率 = 正确保留事实数 / 全部原文事实数；新增不支持率 = （冲突+原文不支持的候选断言）/ 全部候选断言。
含糊事实不移出分母，另列数量。30 条审核不能冒充 100 条自动事实正确率。

## 4. 可选 BERTScore：只在本地新虚拟环境

标准库准备与人工审核无需安装。下面是**用户可选本地命令**，不要在学校服务器已有 Conda 中执行：

```bash
/Users/a1-6/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m venv .venv-semantic-v13
.venv-semantic-v13/bin/python -m pip install 'bert-score==0.3.13' 'transformers>=4.30,<5' 'huggingface-hub<1'
.venv-semantic-v13/bin/python -B v10_extension/docci_semantic_eval_v13/download_evaluator.py \
  --allow-network --output pretrained/eval_longformer_v13
.venv-semantic-v13/bin/python -B v10_extension/docci_semantic_eval_v13/score_bert.py \
  --prepared experiment_results/v13_semantic_local_attempt02 \
  --model-path pretrained/eval_longformer_v13 \
  --output experiment_results/v13_bertscore_smoke_attempt02 --smoke --max-minutes 20
```

下载仅从固定官方模型仓库解析一次 commit，记录 SHA、逐文件哈希；不上传恢复文本到外部服务。
同一准备工作只复用这个冻结的 evaluator receipt，勿根据结果换 revision。版本需连同 pip freeze 保存。
真正评分强制离线、CPU；原文仅作评价参考，绝不进入接收器。模型初始化不计入评分循环超时，预算是批次间检查，不是硬实时中断。
smoke 仅 10 个候选，状态为 SMOKE_ONLY；通过后再用新输出跑完整 1100 个：

```bash
.venv-semantic-v13/bin/python -B v10_extension/docci_semantic_eval_v13/score_bert.py \
  --prepared experiment_results/v13_semantic_local_attempt02 \
  --model-path pretrained/eval_longformer_v13 \
  --output experiment_results/v13_bertscore_full_attempt02 --max-minutes 120
```

真实下载/模型推理尚未在本地执行，不给出未经测量的运行时间保证。先以 smoke 实测再估算，不启动新的训练。

## 5. 对照与结论

shuffled 是历史真实接收输出；错配原文只是评价器参照。原生 BART 未压缩、未传输，是诊断而非理论上限。
数字基线 **NOT_RUN**；不把原文自比较冒充同资源数字满分。未来须匹配符号/功率/带宽/开销/损伤/失败处理。
本轮只有 clean，40dB 是历史哨兵；不能证明 FSO 低 SNR 抗噪、数字悬崖或图文互促。
所有 bootstrap 为描述级、固定种子42、2000次；同一训练seed、反复用过的 qual_dev，不含训练随机性或独立 test 保证。

参考：[BERTScore](https://arxiv.org/abs/1904.09675)、[官方源码0.3.13](https://github.com/Tiiiger/bert_score/tree/v0.3.13)、
[Longformer官方模型说明](https://huggingface.co/allenai/longformer-base-4096)、[SPICE](https://panderson.me/spice/)。
SPICE 仅启发事实拆分；本版未安装 Java/Stanford 解析器，也未声称实现 SPICE。
