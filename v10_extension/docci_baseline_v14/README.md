# V14：单分支基线校准与充分训练

状态：代码与本地非Torch合同已验证；真实BART/VAE/GPU效果待服务器运行。不是新最优模型。保留所有V6—V13源码、权重、原始结果和QUALITY_STOP，不升级现有环境，不自动启动FSO或联合训练。

## 这轮具体改什么

1. **不训练的校准**：同100条qual_dev，对实际V12 CE+KD文字权重使用贪心/部署两种解码；对原生VAE和旧I2使用同图同有效区域评价。原生BART参照是原始分段memory+mask的特权诊断，不是传输方案。
2. **文本两组公平训练**：`legacy64`为现有V4/V10编解码器；`released64`独立实现官方公开代码的窄查询/位置嵌入约定，输出仍为32×64。相同完整描述、九段、CE、批次顺序和优化预算。BART冻结，不使用CLIP、KD或原文接收旁路。
3. **图像两组等更新控制**：同一个已批准I2权重分别继续训练`data64`和`expanded`。更新次数、学习率、架构、像素优先损失相同，仅训练数据集合不同。这样能区分数据扩展与多训练的效果。旧权重不覆盖。
4. 支持全量9647条DOCCI train、梯度累积、磁盘缓存与有界内存；仅原始完整描述，无截断，无随机图文错位裁块。
5. 正确/错配/清零及源骨干诊断、实际文字与PNG、逐帧身份、训练记录、独立CPU验收、私有结果网页和源事实优先的人工盲审。

### 重要文献纠正

LD4LG论文Table13写Perceiver内部768维，但公开源码`0bf9381e049ff288e5e79edc38e4a952a371bee2`的BART路径默认使用`dim_latent=64`、`legacy=False`。所以不能说你的64维必然偏离官方代码。

`released64`是离线独立实现、在本工程完整分帧和严格接收边界下的公开代码对齐对照，**不是官方脚本/环境/训练数据/论文BLEU的完整复现**。保留现有注意力/重建基础模块，训练缓存、初始化与运行协议也未宣称逐位等价。`wide768`是可选容量假设，增加参数/计算量但不增加载荷，不能称官方默认，也不与其宣称等参数。

论文的BLEU97.6不是本项目的WER或光通信正确率。两种推理解码分别记录，部署口径不能替代无约束贪心过旧门槛。

## 1. 上传、解压和检查

将代码包和校验文件上传到现有`/data1/zw/WYH/multimodel`。这个包直接包含`v10_extension/docci_baseline_v14/`，**没有额外的下划线安装目录，也不需要install.py**。

```bash
cd /data1/zw/WYH/multimodel &&
sha256sum -c v10_docci_baseline_v14.tar.gz.sha256 &&
test ! -e v10_extension/docci_baseline_v14 &&
tar -xzf v10_docci_baseline_v14.tar.gz --keep-old-files &&
/home/zw/anaconda3/envs/torch/bin/python -B v10_extension/docci_baseline_v14/launch.py check
```

目录已存在就停，不删除旧文件。源清单会检查历史依赖；若报某文件哈希不同，发回完整报错，不跳过检查或改清单。`check`要求真实Torch/HF CPU合同，缺依赖会失败，不会把SKIP当通过。只写本项目；不pip、不下载新模型、不停止其他人的任务。

## 2. 建议先运行：不训练的校准

```bash
cd /data1/zw/WYH/multimodel
PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python \
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_baseline_v14/launch.py calibrate \
  experiment_results/v14_calibrate_01 --task both --max-hours 2
```

这个动作不会调用训练函数、不会写新神经权重。文本默认读取已核验V12 CE+KD：
`experiment_results/v10_docci_semantic_probe_v12_01/main/text/B1_ce_kd/train/last.pth`。
图像默认读取已核验V10 I2：
`experiment_results/v10_docci_reconstruction_v10_diag_01/diagnostic_I2_shifted/train_clean/last.pth`。
路径不同，可分别用`--text-parent 实际V12/main目录`和`--image-parent 实际diagnostic_I2_shifted目录`，但内容必须仍匹配批准哈希。不自动换其他权重或重新训练。

两种文本解码：

| 口径 | beams | repetition_penalty | no_repeat_ngram_size |
|---|---:|---:|---:|
| greedy（保留主诊断） | 1 | 1.0 | 0 |
| deployment（单列） | 4 | 1.2 | 3 |

两者都不强制末尾EOS，不读取真实长度、源mask或内容来控制停止。原生memory参照另行标注。

## 3. 开始文本训练：先两组，不同时跑图像

```bash
cd /data1/zw/WYH/multimodel
test ! -e experiment_results/v14_text_pilot_01 &&
test ! -e experiment_results/v14_text_pilot_01_launch.log &&
PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python \
nohup /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_baseline_v14/launch.py train \
  experiment_results/v14_text_pilot_01 --task text --profile pilot \
  --max-hours 8 > experiment_results/v14_text_pilot_01_launch.log 2>&1 < /dev/null &
```

默认2048篇完整DOCCI训练描述，20轮，每个微批1篇、累积8次，**每组5120次优化更新**。每篇全部九帧仍参与编码和计费；九帧不是重复发送九次。CE按整个优化步的有效目标token数量加权，最后不足8篇的尾批不会丢弃。

顺序：CPU合同 -> 实际骨干/8篇/2步smoke -> 独立smoke验收 -> 独立新进程从原始初始化开始两组训练。smoke拟合权重不会用于正式训练。不按qual_dev选best、不看到结果后自动加步数。

若明确要测容量，再单独新运行增加`--text-arms legacy64,released64,wide768`。不要初次就把所有选项堆在一起。

全量训练是显式`--profile full`，使用9647篇，默认每组24120更新；不是首次推荐命令。若更改轮数/批量，复制config到项目内新文件，修改后用`--config`传入，并保留原文件。

## 4. 文本完成后，独立运行图像数据量对照

```bash
cd /data1/zw/WYH/multimodel
test ! -e experiment_results/v14_image_pilot_01 &&
test ! -e experiment_results/v14_image_pilot_01_launch.log &&
PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python \
nohup /home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_baseline_v14/launch.py train \
  experiment_results/v14_image_pilot_01 --task image --profile pilot \
  --max-hours 8 > experiment_results/v14_image_pilot_01_launch.log 2>&1 < /dev/null &
```

两组从同一I2权重开始：64图控制 vs 2048图扩展；每组5120更新、10240图次、累积2。训练目标保持`pixel_MSE + 0.01*latent_MSE`，冻结VAE但保留经VAE解码传向codec的梯度。原始I2不是新组训练权重，也不会覆盖。

`--profile full`将扩展组改为9647图、默认5轮、每组24120更新；64图控制仍有相同更新预算。全部原生完整画面512 letterbox，无额外裁块。只访问官方train/qual_dev，不训练或选模用test/qual_test。

## 5. 信道、资源和显存

- 这一轮全部为理想无噪声，**SNR=null/不适用**，没有FSO。旧I2条件输入40仅是既有clean哨兵，不能称接收40dB。
- 文本18432实载荷/9216复载荷；图像8192实载荷/4096复载荷；R=1；无Common、原文UTF8、免费长度头或隐式重复。不是相同比特压缩比，也未测完整光波形开销/BER。
- 使用物理GPU1；启动需16GiB空闲，运行保留4GiB空闲，不会挤掉别人的任务。不是16GiB峰值保证；实际峰值必须看smoke。
- 扩大样本数主要增加磁盘、缓存与训练时间，不自动增加GPU微批。默认LRU仅16条缓存驻留RAM，不将全量BART特征一次堆进内存。
- 现有Python3.8/Torch1.12/Transformers4.30.2环境；FP32无AMP。支持上限48小时是明确停止预算，不是自动续跑或ETA。

时间粗估：校准可能十几至几十分钟；两组文本约1—3小时、图像pilot约1—4小时，**未经本版服务器实测**。缓存、完整双口径生成评价、验收另外计时；日志每20更新给训练阶段ETA，不能把ETAtrain=0当整个任务完成。全量先看pilot实测再定。

## 6. 查看进度与失败处理

文本训练：

```bash
cd /data1/zw/WYH/multimodel
tail -n 60 experiment_results/v14_text_pilot_01_launch.log
tail -n 40 experiment_results/v14_text_pilot_01/main.log
```

缓存阶段会显示`CACHE ...`；训练阶段显示组名、update、unique、loss和ETAtrain；评价每10篇显示进度。完成标记是外层`LAUNCH_FINISHED.json`，不是单条`EVAL`或最后训练step。

报错保留`FAILED.json`、smoke/main日志、learning_audit。不要删失败记录、不重复向同目录启动、不修改MANIFEST、不把旧smoke当正式checkpoint。超时只停止当前launcher直接创建的子进程，无模糊pkill。收集是单独CPU步骤，无旧180秒硬截断。

## 7. 收集发回来（不重新训练）

文本：

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_baseline_v14/collect.py \
  experiment_results/v14_text_pilot_01/main \
  experiment_results/v14_text_pilot_01_records.tar.gz --max-minutes 30
```

图像把路径中的`text`换成`image`。校准用`v14_calibrate_01/calibration`作输入。若同名包/`.contents`已经存在，使用`_records_collect02.tar.gz`，不能覆盖旧包。

结果包含实际图文、summary、index.html、帧、源码快照和验收记录，**是私有材料，不能上传公开GitHub**。大权重与缓存留服务器。工程验收会独立重算文本指标、核对逐帧/错配、训练样本及更新顺序、冻结骨干与源码身份；它不等于恢复质量达标，也不重新执行所有神经推理。

## 8. 可选LPIPS：严格离线，不自动安装

默认明确报告`NOT_MEASURED`，不伪造分数。若已有隔离环境中的`lpips`包及可信官方AlexNet和LPIPS线性权重，建立项目内JSON：

```json
{
  "alexnet": "alexnet-owt-7be5be79.pth",
  "alexnet_sha256": "填写已独立核验的官方整文件SHA256",
  "linear": "alex.pth",
  "linear_sha256": "填写已独立核验的LPIPS线性权重SHA256"
}
```

传入`--lpips-assets 配置路径`。代码只在这个显式选项下导入LPIPS；两种自动预训练下载均关闭，先核对本地权重再装入。缺包/缺权重就报错，不pip或换网络。不要只为这项指标升级原torch环境。LPIPS使用全部已评价图像的有效矩形，而非只挑保存PNG样例。

## 9. 人工事实审核

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_baseline_v14/audit.py prepare \
  experiment_results/v14_text_pilot_01/main experiment_results/v14_text_facts_01
```

先仅看原文，将`source_facts.jsonl`中每篇的`facts`填写为`[{"id":"F1","text":"一个具体事实","visually_verifiable":true}]`并设reviewed=true。事实可含对象、数量、颜色、动作、关系，也要保留不可视觉推断的文本私有事实。不要先看哪个模型恢复得好再挑事实。

然后：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_baseline_v14/audit.py blind \
  experiment_results/v14_text_pilot_01/main experiment_results/v14_text_blind_01 \
  --facts experiment_results/v14_text_facts_01/source_facts.jsonl
```

将blind_review交给审核者，PRIVATE_KEY不要给。逐条事实选择retained/contradicted/omitted/unclear，填写原文不支持的新增断言和generated_claim_count，设reviewed=true。不能修改生成原文。每30篇对应多组候选，不是只审核30条输出。

```bash
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_baseline_v14/audit.py score \
  experiment_results/v14_text_blind_01 experiment_results/v14_text_facts_scores_01.json
```

未完成矩阵拒绝评分；unclear仍留分母。原文不支持新增不等于视觉上一定为假。单人审核、单训练种子不包含评审者/训练随机性不确定性。V13已有评分与人工协议不改写，新审核结果另存。

## 边界

本版尚未实现像素域SwinJSCC训练、BPG/真实FEC数字对照、新跨模态融合或FSO带噪训练；这些不是本版测试通过就自动完成的事项。先根据单分支对照和事实审核决定是否进入下一阶段。没有论文发表概率或恢复必胜保证。

详细文献与实现差异见[SOURCES.md](SOURCES.md)，本地验证见[VERIFICATION.md](VERIFICATION.md)。
