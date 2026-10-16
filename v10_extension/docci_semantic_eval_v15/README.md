# V15：先核实语义，再改变文本模型

这是独立、只读的评价版本，**不是新训练模型**。保留V6—V14代码、权重、原始输出、WER和QUALITY_STOP。默认仅Python3.8标准库，不用GPU、不联网、不安装依赖，不加载上传records中的代码或pickle。

## 已实现与尚未实现

- 原V13完整100篇/1100条输出接入，独立重算字面指标；真实错配接收、清零、错配原文、原生BART分别展示。
- 已有Longformer BERTScore严格复用：核对原prepared身份、完整矩阵、模型/版本/层/设置和组均值；10条smoke不可混作正式结果。不重复推理。
- “相对参照位置”是`(正常均值−错配均值)/(原生均值−错配均值)`，**不是恢复准确率**；不强行截到0—1，区间用配对描述重采样。原生不是严格上界。
- 40条手工构造的合成挑战（四个虚构场景），检查数量、颜色、属性绑定、空间关系、否定、动作、遗漏、无关描述和同义改写。它们不是模型输出或DOCCI实测结果。
- 原先按长度选好的30篇人工审核不重选；再按源长度划15篇开发/15篇确认。两者均来自反复使用的qual_dev，不是新模型测试集。完整V13审核对应330条候选，而不是只看30条输出。
- 源事实先定义、冻结后再看随机编号的候选；保留物体/属性/数量/动作/关系/否定等事实类型与不确定项分母。审核未做不能写“事实恢复通过”。
- 可选读取V14已完成的文本main或calibration结果，保留greedy/deployment分轨。仅核对light文件哈希、文本矩阵和WER；不宣称重新验证神经前向、全部帧或训练历史。已包含的BERTScore仍仅覆盖原V13输出，V14明确NOT_MEASURED。
- CAPTURE/AlignScore是**冻结协议与完整逐行分数导入接口**，不是已实现的离线模型适配器。没有下载权重、安装包、CPU/GPU兼容性或真实分数验收。必须先审核上游依赖与本地资源，再单独决定是否接入；不要在原Conda中直接pip。

主指标仍是人工关键事实保留率与原文不支持的新增事实率；BERTScore辅助。不能把替换错误一律叫近义替换，也不能把AlignScore的`1-score`称矛盾率。CAPTURE数量和属性绑定局限见[SOURCES.md](SOURCES.md)。

## 1. 上传代码、校验、解压

只把源码包和校验文件上传到现有项目。直接展开到模块，无下划线安装器：

```bash
cd /data1/zw/WYH/multimodel &&
sha256sum -c v10_docci_semantic_eval_v15.tar.gz.sha256 &&
test ! -e v10_extension/docci_semantic_eval_v15 &&
tar -xzf v10_docci_semantic_eval_v15.tar.gz --keep-old-files &&
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_semantic_eval_v15/run.py check &&
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_semantic_eval_v15/test_eval.py
```

已有目录就停止，不删除，不改MANIFEST。测试只用合成数据和合成评分，不表示CAPTURE/AlignScore/真实恢复已经通过。

## 2. 只读现有文本（不重新训练）

用户已有服务器V13离线包，其prepared路径如下；若移动过，替换为实际原目录：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_semantic_eval_v15/run.py prepare \
  --prepared _v13_server_cpu_offline_01/prepared \
  --output experiment_results/v15_semantic_review_01/prepared
```

此步骤只需小量RAM；不调用Torch，不需要新权重。输出网页`prepared/index.html`展示完整恢复，`prepared/source_review.html`仅展示原文供事实清单阶段使用。默认用新目录，重复运行改`_02`。

**若本地已经交付同一版本的私有已处理records，可以直接看它，不必重复执行。**源码包不含原文；私有结果包不能公开上传GitHub。

原有完整BERTScore已在服务器的哪个目录，以真实路径为准，可先只读查找：

```bash
find experiment_results -maxdepth 3 -type f -name bertscore_rows.jsonl -print
```

把查到的路径发回来后，才能给出该服务器上的准确导入命令；不要猜完整评分目录、不要选smoke。程序入口为`metrics.py import-bert`，需要指定prepared、scores、summary、output四个参数，`--help`可查看接口。此步骤已在本地对用户回传的完整评分执行，私有结果可以直接查看，无需为了重分析在服务器再跑一遍。

完整100篇的描述级区间只包含这些固定输出的抽样不确定性；单训练种子、qual_dev复用、没有事实审核，不作确认性显著性或质量达标结论。不会悄悄打开缺少官方基线表的Longformer重缩放。

## 3. 可选：只算40条评价器挑战，不重新算1100条

复用服务器已经能用的CPU叠加环境和官方Longformer权重，不再下载：

```bash
cd /data1/zw/WYH/multimodel
HF_HOME=/data1/zw/WYH/multimodel/.cache/v13_hf \
MPLCONFIGDIR=/data1/zw/WYH/multimodel/.cache/v13_mpl MPLBACKEND=Agg \
.venv-v13-eval-cpu/bin/python -B -u \
  v10_extension/docci_semantic_eval_v15/score_challenge_bert.py \
  --prepared experiment_results/v15_semantic_review_01/prepared \
  --model-path _v13_server_cpu_offline_01/model \
  --output experiment_results/v15_semantic_review_01/challenge \
  --max-minutes 20
```

启动核对全部权重哈希，CPU四线程、batch1；完整文本预检，超长直接失败，不截断。日志每条显示`BERT challenge x/40`。预计分钟级只是规划，未实测；20分钟为停止预算而非ETA。它检查评价器会不会偏爱错误细节，**不改善训练模型，也不产出事实正确率**。未实际跑不能写“挑战已通过”。

## 4. 人工事实审核：先原文，后候选

先看`source_review.html`，编辑模板并另存为`source_facts_completed.json`（不要改冻结prepared内的模板），填写：

```json
{
  "schema": "v15-source-facts",
  "inventory_reviewer": "审核者代号",
  "examples": [{
    "example_id": "保持模板内容",
    "reference": "保持原文",
    "reference_sha256": "保持模板哈希",
    "inventory_complete": true,
    "facts": [{
      "fact_id": "F1", "category": "count",
      "proposition": "一个可单独判断的数量事实",
      "reference_quote": "必须逐字摘录原文中的支持片段",
      "visually_verifiable": true
    }]
  }]
}
```

全部30篇都要填写；事实类型见protocol。文本独有、无法从图片推断的事实也要保留，不能只挑容易恢复的对象名。

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_semantic_eval_v15/audit.py blind \
  --prepared experiment_results/v15_semantic_review_01/prepared \
  --facts experiment_results/source_facts_completed.json \
  --output experiment_results/v15_fact_blind_01
```

仅把`candidate_review_template.jsonl`和审核说明交给候选审核者，**不交PRIVATE_KEY、模型结果网页或组别映射**。另存完成版judgements.jsonl。source事实逐项标`entailed/contradicted/not_mentioned/unclear`；前两种附候选精确引句。prediction_facts逐条列出模型输出里的原子断言，标`supported/contradicted/unsupported/unclear`并附候选引句；supported/contradicted还需原文引句。填写reviewer和prediction_inventory_complete=true。非空输出不能用空预测清单骗出“零幻觉”。

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_semantic_eval_v15/audit.py score \
  --prepared experiment_results/v15_semantic_review_01/prepared \
  --audit experiment_results/v15_fact_blind_01 \
  --judgements experiment_results/judgements.jsonl \
  --output experiment_results/v15_fact_scores_01
```

不完整或选择性审核拒绝汇总。15+15分别报告；不自动挑最有利的评价器，不在确认子集上反复调参。空预测的事实保留为0，新增事实率未定义并单列数量；unclear留分母。原文不支持≠图像中一定为假。330条长描述的审核需要人工工作，不能承诺几分钟完成。

## 5. 将来接V14文本：不重训、不偷换协议

V14文本两组完成并独立收集后，用实际解包records根目录或原main目录：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B -u v10_extension/docci_semantic_eval_v15/run.py prepare \
  --prepared _v13_server_cpu_offline_01/prepared \
  --v14-root experiment_results/v14_text_pilot_01/main \
  --output experiment_results/v15_with_v14_text_01/prepared
```

原V13 BERTScore不能冒充新增V14分数；新候选组要另行完整评分和盲审。当前无V14新文本训练结果回传，不能声称released64优于legacy64。下一步训练命令见[RUN_NEXT.md](RUN_NEXT.md)。

## 6. CAPTURE / AlignScore：冻结分数接口，不是自动安装器

先用隔离环境完成真实运行兼容性和离线资源审计。本版**不会执行**这两个模型；不得把接口测试通过写成真实模型评分通过。第三方准备好后，先将metric、evaluator_revision、settings与实际本地assets的path/sha256写入spec.json，再freeze：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_semantic_eval_v15/metrics.py freeze \
  --prepared experiment_results/v15_semantic_review_01/prepared \
  --spec experiment_results/metric_spec.json --output experiment_results/v15_metric_plan_01
```

外部评分逐行输入`requests.jsonl`，每行输出：case_id、reference_sha256、candidate_sha256、plan_sha256（plan.json整文件SHA）、score（有限实数）、details（对象）。所有真实输出和40挑战均须完整，不可丢失败或空输出；无法解析要保存现场而非编造0分。不能先算分再冻结“预先”协议。

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_semantic_eval_v15/metrics.py import \
  --plan experiment_results/v15_metric_plan_01 --scores experiment_results/external_scores.jsonl \
  --output experiment_results/v15_metric_import_01
```

导入仅验证标量与身份，不独立证明第三方执行正确。CAPTURE不保证发现数量和颜色归属错误；AlignScore不表示校准矛盾率，也不能衡量遗漏。不要用评价器高分代替关键事实人工证据。

## 7. 收集结果

```bash
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_semantic_eval_v15/collect.py \
  --prepared experiment_results/v15_semantic_review_01/prepared \
  --bert experiment_results/v15_semantic_review_01/bertscore \
  --output experiment_results/v15_semantic_review_01_records.tar.gz
```

如已运行挑战可加`--challenge experiment_results/v15_semantic_review_01/challenge`。没有BERT重分析目录就省略`--bert`。只收这几项冻结结果和本版源码快照，不收模型权重、不重新训练；包含原文，**只能私下返回，不能公开GitHub**。同名存在换collect02。

本版不修改训练结构、损失、数据规模或原门槛，不自动启动学校GPU，不声称连续优于数字或图文互促。新代码公开、原始图文私有；新结果先看事实错误类型，再决定扩数据或cross-attention LoRA的单变量实验。
