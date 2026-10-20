# V28 独立连续 token：低 SNR 训练分支

本分支停止图文互促，复用V14纯图像codec、V18独立文本codec和原生特征缓存。没有新模型下载、没有额外依赖安装、没有覆盖历史代码。目标是验证**训练SNR覆盖负值之后，单次连续传输能否更抗噪**，不是已经证明效果更好。完整规则见[PROTOCOL.md](PROTOCOL.md)。

## 1. 上传与解压

将 `v10_docci_direct_token_v28.tar.gz` 和同名 `.sha256` 上传到服务器 `/data1/zw/WYH/multimodel/`，在服务器执行：

```bash
cd /data1/zw/WYH/multimodel &&
sha256sum -c v10_docci_direct_token_v28.tar.gz.sha256 &&
test ! -e v10_extension/docci_direct_token_v28 &&
tar -xzf v10_docci_direct_token_v28.tar.gz --keep-old-files
```

这是直接解压到 `v10_extension/` 的增量包，**没有install.py**。若目录已经存在，不删除、不覆盖；先核对是否是这次同一代码包。依赖已有项目V14/V18，不是独立的新项目。

## 2. 只读检查位置，不占GPU

```bash
cd /data1/zw/WYH/multimodel
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_direct_token_v28/launch.py inventory --task both
```

默认复用：

- 图像缓存：`experiment_results/v14_image_pilot_02/main/image/`，权重为其 `expanded/train/last.pth`。
- 文本缓存：`experiment_results/v14_text_pilot_01/main/text/`，权重 `experiment_results/v18_text_noise_01/train_noise_replay/last.pth`。
- 本地预训练骨干：`pretrained/sd-vae-ft-mse`、`pretrained/bart-base`。
- 真实回放所需旧rows位于V14 expanded/validation和V18 eval_noise_replay。原始记录不能缺失。

存在不代表校验通过；后续还会检查固定SHA、原生缓存、实际权重和真实CUDA。

## 3. 建议先跑图片pilot

这一步自动执行合同测试、真实小样本smoke，再跑两组正式pilot以及冻结父模型评价。`--task image` **不会启动文本训练或加载BART权重**。首轮只用AWGN，FSO待单模态结果确认后另做，不混淆两种信道。

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
export V28_OUT=/data1/zw/WYH/multimodel/experiment_results/v28_image_low_snr_pilot_01
test ! -e "$V28_OUT" &&
test ! -e "${V28_OUT}_launch.log" &&
nohup "$PYTHON_BIN" -B -u v10_extension/docci_direct_token_v28/launch.py \
  pipeline "$V28_OUT" --task image --profile pilot --gpu 1 --max-hours 12 \
  --lpips-assets /data1/zw/WYH/multimodel/_v20_lpips_assets \
  > "${V28_OUT}_launch.log" 2>&1 < /dev/null &
```

若 `_v20_lpips_assets` 不存在，可**删除上面整行 `--lpips-assets ...`**，此时只测PSNR/SSIM，LPIPS明确未测；不要下载/安装替代。原V20已使用过的权重目录可以直接复用。

如只想先看预检：将 `pipeline` 改成 `smoke`，使用新的 `V28_OUT`。smoke只测小样本和真实更新，不代表性能。

显卡若GPU0更空闲，用 `--gpu 0`；程序只设置自己及子进程的环境，不修改全局设置，不抢占/终止其他进程。图片与文字默认顺序跑，避免同时占同一张显卡。

## 4. 看进度和取结果

```bash
tail -n 60 "${V28_OUT}_launch.log"
tail -f "$V28_OUT/pilot/image/train_low_snr.log"
```

当前阶段在 `$V28_OUT/progress.json`，每组训练日志有`更新/总更新`和`ETAtrain`，只包括当前组训练，不包括后面的评价。某阶段还没开始，日志不存在是正常的；先看launch日志。按Ctrl+C退出tail不会停止后台训练。

完成后必须存在 `$V28_OUT/VERIFIED_COMPLETE.json`，结果包在 **`${V28_OUT}_records.tar.gz`**，同目录另有`.sha256`。把这两个发回；大权重/原生缓存/TX数组留服务器。包内网页和文本是PRIVATE，不上传公开GitHub。

若已经训练评价完成、只是收集超时，**不要重训**。使用新包名独立验收收集：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B -u \
  v10_extension/docci_direct_token_v28/collect.py \
  "$V28_OUT" "${V28_OUT}_records_collect02.tar.gz" --full --max-minutes 120
```

如果运行中失败，先收诊断记录（不伪装已完成）：

```bash
/home/zw/anaconda3/envs/torch/bin/python -B \
  v10_extension/docci_direct_token_v28/collect.py \
  "$V28_OUT" "${V28_OUT}_PRIVATE_failed_records.tar.gz" --partial
```

不要重复启动同一个目录，不自动续训、删除失败目录或杀别人的进程。

## 5. 文本及扩大规模

图片pilot检查后，文本pilot换新目录、去掉LPIPS选项，改成`--task text`：

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
export V28_OUT=/data1/zw/WYH/multimodel/experiment_results/v28_text_low_snr_pilot_01
test ! -e "$V28_OUT" &&
test ! -e "${V28_OUT}_launch.log" &&
nohup "$PYTHON_BIN" -B -u v10_extension/docci_direct_token_v28/launch.py \
  pipeline "$V28_OUT" --task text --profile pilot --gpu 1 --max-hours 12 \
  > "${V28_OUT}_launch.log" 2>&1 < /dev/null &
```

`--profile full`为2048样本、10轮、100条qual_dev、3次独立噪声试验。图片每组10240更新、文字每组2560更新；大矩阵自由生成可能很久，必要时`--max-hours 24`或48。`--task both`会顺序做两模态所有流程；首轮不建议直接一起full。

pilot图像每组1024更新，文本每组256更新，二者都另有smoke和三条评价路径。**未实测本版速度**；按旧流程估计图片pilot约1–3小时、文本pilot约0.5–2小时，读盘/评价可能超过估计。12小时是停止上限，不是预计必需时间。以后以smoke实际速度和日志ETA修正。

## 6. 结果怎么解释

主表看 −5dB，`low_snr`是否比相同步数`normal_range`更好，并完整看clean损失、所有SNR和清零/错配对照。图片PSNR/SSIM/可选LPIPS，文字WER/CER与原文接收文本。低SNR准确率不是bit accuracy；连续传输没有直接BER。

即便低SNR有增益，也不能直接宣称超过数字传输、FSO链路有效、联合压缩或图文互促。现有V19/V23数字基线保留，当前不重新跑数字。量化低SNR收益后，再决定是否以新固定权重接入原数字/FSO对照。

## 状态

新增代码可复用旧源文件；Python3.8语法和纯CPU合同在本地核验。此机器无Torch，神经网络合同未本地执行，必须服务器执行；`V28_REQUIRE_TORCH=1`阻止缺失依赖被跳过当成功。新真实权重/CUDA/训练/性能均尚未运行。
