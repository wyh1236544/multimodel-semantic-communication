# V10-A：先补齐信道，再验证共享语义保护

状态：V10-A 的物理/网络实现保留，本版只修订项目内安装、路径隔离与运行入口。部署测试已在本地执行；先前版本的16项数值/合成联调通过不等于本次重新执行。本次原有本地 PyTorch 目录访问受限，未改其权限或重装它，完整数值复测交由服务器现有环境执行。真实 COCO/CLIP/VAE/GPU 性能仍未测。它不是已经超过 V9 的新最优模型。

设计与文献：[docs/DESIGN.md](docs/DESIGN.md)。工程规则：[docs/engineering-habits.md](docs/engineering-habits.md)。交接：[docs/HANDOFF.md](docs/HANDOFF.md)。

正式论文实验统一入口见 [docs/FORMAL_EXPERIMENTS.md](docs/FORMAL_EXPERIMENTS.md) 与根目录 `run_v10_formal_suite.sh`。该入口增加三训练种子、validation 冻结门禁、一次性 test、Common 错配对照、尾部分位数和层级 bootstrap；它不代表这些实验已经执行。

## 1. 安全部署：全部在现有 multimodel 内

本版取代之前“创建并列 multimodel_v10 目录”的部署说明。**只上传 `multimodel_v10_inplace_20260920.pyz` 到 `/data1/zw/WYH/multimodel/`。** `.pyz` 是 Python 可直接运行的代码压缩包，不含数据或权重，不需要 pip 安装。不要用旧的 tar.gz 包，也不要解压到父目录。

在服务器运行：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
"$PYTHON_BIN" -B multimodel_v10_inplace_20260920.pyz --project-root . --dry-run
"$PYTHON_BIN" -B multimodel_v10_inplace_20260920.pyz --project-root .
bash run_v10.sh package
```

新增的代码仅为根目录 `run_v10.sh`、`README_V10_INPLACE.md` 和 `v10_extension/`。旧 `train.py`、`test.py`、根目录 `models/`、`utils/`、配置、V9-B.2 选择器及数据/权重/结果不覆盖。新版的模型副本放在 `v10_extension/models/`，由新版入口独立导入，避免影响旧脚本。要运行旧版，仍在同一个 multimodel 目录运行旧命令。

安装器先检查全部目标的冲突、路径和哈希，再新增文件。同名且内容不同会在写入前停止；相同内容允许重复安装而不改写文件。安装记录位于 `v10_extension/deployment_records/`，清单位于 `v10_extension/PACKAGE_MANIFEST.json`。上传附带的 `.pyz.sha256` 后可用 `sha256sum -c multimodel_v10_inplace_20260920.pyz.sha256` 校验压缩包。

所有以下命令均从 multimodel 根目录运行。日志在 `logs/v10/`；临时文件、模型库缓存在 `.cache/v10/`；新输出只用 `experiment_results/v10_*`；划分清单默认在 `v10_extension/manifests/ln_v10_splits.json`。已有输出拒绝覆盖，项目外写路径和写路径中的符号链接会被拒绝。现有数据目录中的只读链接可继续使用。环境变量只传给此次子进程，不写入 `.bashrc`、全局 Codex 文件或 Conda 配置；不修改已有解释器。

想先只看实际路径、不开始训练，可运行：

```bash
bash run_v10.sh smoke experiment_results/v10_smoke_seed42 --dry-run
```

本轮也不自动删除或回滚之前已经创建的项目外目录/全局文件，因为这同样是项目外修改。

不要执行全环境升级，不要重装旧服务器上的 PyTorch/Transformers/Diffusers。先用现有已运行 V9 的 `torch` 环境。首次真实模型预检若暴露依赖问题，应保存完整错误再定位。

## 2. 先做 CPU 检查与固定划分

```bash
bash run_v10.sh check
bash run_v10.sh splits
```

`check` 不下载模型、不训练真实数据；包含真正的信道数值测试和使用小型替代骨干的端到端代码联调。约数秒至数十秒，取决于 CPU，不能由此估计实际 GPU 训练耗时。

`splits` 检查全部图像存在，固定 Localized Narratives 的图像 ID 和标注哈希。训练集来自 train2017；验证500、校准300、测试1000来自 val2017 中排除已知历史选模图像后的部分。默认排除旧 seed42 前500个 ID，它覆盖我们已查明的选模复用问题。若你还用过其他图像做调参/目视选优，请先补入 JSON ID 列表：

```bash
bash run_v10.sh splits --extra-excluded-json v10_extension/manifests/other_used_val_ids.json
```

此命令应在首次生成划分时使用；既有清单拒绝覆盖。不知道完整历史时，报告只能称“排除了已知历史使用”，不能保证所有图片从未看过。后续不因测试结果不理想而换划分。

## 3. 选择空闲 GPU，然后真实预检

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
```

选择你有权使用的 GPU。下例用物理 GPU 0；若用 GPU 1，改为 `CUDA_VISIBLE_DEVICES=1`，程序内仍用 `cuda:0`。默认不足8 GiB可用显存或8 GiB可用磁盘会停止；这些只是预检下限，不是完整运行资源保证。禁止停止别人的任务。

```bash
CUDA_VISIBLE_DEVICES=0 bash run_v10.sh smoke experiment_results/v10_smoke_seed42 \
  --batch-size 2 --num-workers 0 --seed 42
```

预检固定8张训练图、4张验证图、1轮，并限制30分钟。检查输出目录中 `checkpoint_load.json`、`history.json`、`COMPLETE.json`。脚本会在结束后独立验收。若输出目录已存在，请改成 `_attempt2`，不要覆盖。

日志会及时刷新；前面加载本地 CLIP/VAE/权重和检查文件需要时间，加载阶段没有训练 step 是正常的。程序不会把没有完成的运行标为成功。超时/异常保留当前日志和已保存 checkpoint；本版未实现精确断点续训，重新指定 checkpoint 是一次新的微调实验，不是原运行无缝续跑。

## 4. 第一轮建议：先跑2000张试验

```bash
CUDA_VISIBLE_DEVICES=0 bash run_v10.sh train experiment_results/v10_pilot_seed42 \
  --epochs 2 --train-samples 2000 --val-samples 64 \
  --batch-size 2 --grad-accum 2 --num-workers 0 --seed 42 --max-hours 3
```

第一轮只判断：能否稳定训练；FSO/Rayleigh 是否相对零微调改善；Common-only 是否比随机 Common 有效；高 SNR 是否严重退化。不要预先把它当论文正式结果。

若这轮成功，再做6轮/10000张，验证扩大到500：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_v10.sh train experiment_results/v10_mixed_seed42 \
  --epochs 6 --train-samples 10000 --val-samples 500 \
  --batch-size 2 --grad-accum 2 --num-workers 0 --seed 42 --max-hours 12
```

默认从 `experiment_results/ln_v9a_feedback_generation_seed42/seed42.pth` 初始化，不覆盖它，也不自动接着2000张实验继续。若你的 V9-A 权重实际存放位置不同，先确认文件，再用 `INITIAL_CHECKPOINT=experiment_results/实际目录/seed42.pth` 指定。不要把 V9-B.2 verifier 权重作为主网络权重。若从 V10 checkpoint 微调，必须保持划分清单一致。不要拿 smoke 权重作为正式主初始化。

实际时间取决于 GPU 空闲、I/O 和模型。新增 Common-only 与 AWGN 回放增加计算，FSO 随机采样不是主要开销。先用预检和2000张运行记录评估成本；此处12小时是停止预算，不是预测耗时。时间到则标记未完成，不自动无限续跑。

## 5. 先在 validation 上做可配对对照

固定相同图像、种子、信道/SNR/重复数。先看旧 V9 权重零微调进入新协议的结果（不是原V9运行分数）：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_v10.sh eval \
  /data1/zw/WYH/multimodel/experiment_results/ln_v9a_feedback_generation_seed42/seed42.pth \
  experiment_results/v10_zero_shot_val \
  --split validation --samples 100 --repeats 3 --save-images 8

CUDA_VISIBLE_DEVICES=0 bash run_v10.sh eval \
  experiment_results/v10_mixed_seed42/best.pth experiment_results/v10_mixed_val \
  --split validation --samples 100 --repeats 3 --save-images 8

bash run_v10.sh compare experiment_results/v10_zero_shot_val \
  experiment_results/v10_mixed_val experiment_results/v10_comparison_val.json
```

默认5种信道×5个SNR×3次重复×100张=7500次独立逐图前向，不含扩散。比你以前100张/单次/单信道评估大很多。初次可增加 `--profiles awgn fso_moderate --snr-list 0 10 --samples 16 --repeats 1` 缩小规模；两份对照要用同样参数。

每图每信道噪声由固定身份生成，不依赖加载顺序。输出：`samples.jsonl`（包含接收文本/Common短描述）、`summary.json`（分组指标、检索、开销）、`images/`、运行身份与完成证据。

图像命名：`imageID__profile__snrX__rY__ablation.png`，其中 X 是名义 SNR，Y 为独立信道重复索引；`__source.png` 仅为参考图，不用于接收端生成提示。

## 6. 真正只传 Common 的效果

```bash
CUDA_VISIBLE_DEVICES=0 bash run_v10.sh eval \
  experiment_results/v10_mixed_seed42/best.pth experiment_results/v10_common_only_val \
  --split validation --samples 100 --profiles rayleigh fso_moderate fso_strong \
  --snr-list -5 0 10 --repeats 3 --ablation common_only

CUDA_VISIBLE_DEVICES=0 bash run_v10.sh eval \
  experiment_results/v10_mixed_seed42/best.pth experiment_results/v10_random_common_val \
  --split validation --samples 100 --profiles rayleigh fso_moderate fso_strong \
  --snr-list -5 0 10 --repeats 3 --ablation common_only_random
```

这会发送 C32，不发送 I/T。重点看 Common→文本检索、`common_caption`、Common语义相似度和错误类型。图像也保存，但不要求恢复精确原图。随机 Common 保留相同载荷和平均功率，作为语义信息对照；检索图库100时，随机排名的期望 R@1 为1%，但实际随机网络输出可能存在偏置，需要实测。

## 7. 可选扩散预览：不与训练同时占显存

先退出训练/评估进程，使用其保存的接收图与接收文本：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_v10.sh generate \
  experiment_results/v10_mixed_val experiment_results/v10_diffusion_preview \
  --strength-list 0.25 0.35 0.45 --max-candidates 16 --max-minutes 30
```

只有保存了图像、前向头未被拒绝、接收模式为 hybrid/generation、接收提示非空的样本会生成。`--max-candidates` 为总候选上限。提示不读原文。此步骤不加载旧 V9-B.2 verifier，也不自动宣称选出的图更好；只是接收信息驱动的候选预览。它不能证明扩散增益，正式强度/安全阈值必须在 calibration 划分重新校准。

## 8. 必须补齐的论文实验

先用同样6轮/10000张/验证设置分别训练 `--regime awgn`、`--regime fso_moderate`、默认 mixed；不能把单信道训练次数减半后说混合模型更好。

然后分别训练 `--no-uep`、`--no-adapter`、`--no-common-probe`、`--keep-early-fusion`，每次只改一项。评估这些模型时要带匹配设置；配置变化会检查，临时诊断需明确加 `--allow-settings-change`。不重新训练的关模块只反映该 checkpoint 的依赖，不是公平方法对照。

理想CSI上界可在同一checkpoint评估时增加 `--csi perfect --allow-settings-change`；ZF对照为 `--equalizer zf --allow-settings-change`。多块分集需分别训练/评估 `--blocks 4`，并给均等资源的非交织对照，不能把一块慢衰落当成四块独立链路。

方法与阈值冻结后，再在 test 上评估1000张，每图3次或更多信道重复，至少3个训练种子。最终需要独立语义评价/人工盲评、尾部/失效概率、总资源、实际运行耗时。详细假设和未实现物理因素见设计正本。

## 9. 发回哪些文件

先发真实 smoke 的日志、`checkpoint_load.json`、`history.json`；正式对照完成后再发两个评估目录的 `summary.json`、`samples.jsonl`、比较报告，以及少量同imageID面板。不要上传大权重和数据集。
