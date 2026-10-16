# V15现有任务接入保护：不重复训练

2026-10-16。用户确认正在运行`experiment_results/v14_text_pilot_01`的V14文本pilot。
**让它继续，别再次执行V14 launch.py train，也别另建pilot02重训。**本说明取代旧V15 RUN_NEXT里对当前这次任务的“尚未启动则训练”建议。用户报告运行中，不代表本地代理已检查服务器PID或剩余时间。

新模块只做完成检测与一次性只读接入。V14/V15原有源码、manifest、权重、记录保持原样；无Torch、无GPU、无子进程启动、无网络、无安装依赖、无自动等待/轮询。未增加CAPTURE/AlignScore或新语义分数。

## 1. 安装（正在训练时也可以）

新包自带原版V15源码与独立guard；因此不要求先安装旧V15包。安装器预检所有冲突与依赖，只添加缺失且已知的源码；原V15已存在且一致时不改文件或时间戳，未知编辑直接拒绝。**绝不写docci_baseline_v14或experiment_results。**旧包/旧V15结果不用删除。

把新包和sha256上传到现有项目根目录，然后执行：

```bash
cd /data1/zw/WYH/multimodel &&
sha256sum -c v10_docci_semantic_eval_v15_guard.tar.gz.sha256 &&
test ! -e _v10_docci_semantic_eval_v15_guard &&
tar -xzf v10_docci_semantic_eval_v15_guard.tar.gz --keep-old-files &&
/home/zw/anaconda3/envs/torch/bin/python -B \
  _v10_docci_semantic_eval_v15_guard/install.py --project-root . &&
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_semantic_eval_v15_guard/guard.py check
```

若下划线解压目录已存在，不重复解压，不删除；可以直接再次执行同一install.py，它会核对并跳过内容一致的文件。安装不修改正在运行的V14程序。

## 2. 现在：只查看是否可接入

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_semantic_eval_v15_guard/guard.py status
```

固定默认目标就是`experiment_results/v14_text_pilot_01`。不扫描、猜选或启动别的训练：

- `NOT_READY`（退出码2）：外层完成标记没有出现，可能仍在运行，也可能已中止；本命令不检查进程活性、不开始评价，也不创建输出目录。**不是要求重训。**
- `FAILED_PRESERVED`（退出码3）：有失败记录，保留现场，不自动重启。
- `READY_FOR_READ_ONLY_IMPORT`（退出码0）：外层LAUNCH_FINISHED、main完成/源身份、原CPU验收日志均对应这次完整两组text pilot。不是质量通过。

未完成时先让现有任务继续；需要判断是否仍活着，再结合原launch/main日志和精确进程信息。最后update、main/RUN_FINISHED或单独某组EVAL都不能代替外层完成标记。

## 3. 完成以后：同一条接入命令只处理一次

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_semantic_eval_v15_guard/guard.py prepare
```

原V13输入默认`_v13_server_cpu_offline_01/prepared`。输入丢失时只报告错误，不重新生成、更不训练。
程序检查完成状态后，才读取真实V14结果，复用原V15的文本适配器。全部请求使用唯一确定目录：

```text
experiment_results/v15_from_v14_text_pilot_01/
├── IMPORT_STARTED.json
├── prepared/index.html
├── prepared/source_review.html
├── prepared/cases.jsonl
└── IMPORT_FINISHED.json
```

重复执行同一命令：同一组完整输入已接入时返回`ALREADY_IMPORTED_NO_REPEAT`，仅校验冻结结果，不再次生成、训练或计算BERTScore。两个调用同时进入时，原子目录预约只允许一个写；另一个返回`OUTPUT_RESERVED_OR_PARTIAL`（退出码4）。失败/中断也保留这个状态，不自动清理或重试；把日志发回排查，不删除目录重来。

接入包含原V13的1100输出及V14两组+原生、两种解码的1400输出，共2500条。**没有重复神经推理**，只是保存文本与字面指标重算；V14的BERTScore明确NOT_MEASURED，不能拿旧V13分数冒充。旧QUALITY_STOP和人工事实PENDING保留。

此入口不重新运行V14全量帧/训练历史验收，而是核对原launcher成功状态和已保存CPU验收日志；随后V15适配器再检查全部light文件哈希、完整文本矩阵和WER。它不是硬件证明或新的模型质量门槛。

## 4. 发回原V14结果（不再训练）

等V14完成，只收集当前这次运行：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_baseline_v14/collect.py \
  experiment_results/v14_text_pilot_01/main \
  experiment_results/v14_text_pilot_01_records.tar.gz --max-minutes 30
```

如同名包或.contents已有，保留旧包，换collect02，不重训。上传给代理的小records包含原始图文，不能公开GitHub；权重留服务器。

## 验证边界

test_guard.py是明确合成的状态、日志和计数：未完成/只有main完成/失败/错误任务/错源/验收不完整、部分残留、重复调用、并发、输入改变、异常保留，以及安装冲突/幂等保护。不能据此声称用户当前V14已完成或真实文本恢复提高。

源码包仅代码，带整包SHA与payload逐文件SHA。原V15所有文件字节不变；修改只在新guard与共享当前状态说明。没有新训练、权重下载、全环境升级、长期监控或后台任务。
