# 导频自适应模式分配增益 V1

状态：新增隔离实验代码，非训练升级。真实 COCO/GPU 测量尚待服务器执行。
只增加 `v10_extension/pilot_gain/`；不修改旧模型、信道、权重、划分或结果。

## 目的与边界

固定同一 mixed checkpoint，对比 adaptive / fixed_reconstruction(0) /
fixed_hybrid(1) / fixed_generation(2)。只改变“导频可靠性是否决定模式分配”，
保留导频估计、MMSE、UEP、接收门控、反向 RF 模式码和前向模式头。
固定策略在噪声反馈后仍可能出现实际发送模式偏差；统计 sender_mode，不能假装反馈无误。
这不是“有导频 versus 无导频”的总收益，也不是固定策略各自重新训练后的最优对比。
模式切换会同时改变 C/I/T 配额和既有模式 UEP 表，因此结果属于整套模式策略，不只配额。

原始连续信号 RF/IMDD 模型，**不是 physical_link V2 相干 QAM 波形**。
全量4608实特征；AWGN前向2400复符号，FSO前向9408光强时隙，反向64实值RF符号单列。
不把这些连续值折算为字节，不虚构 BER 或实际光链路传输时间。
generation 是分配名称，不执行扩散。保存重建图、收到的文本和 Common 描述。

默认仅用一份 seed42 训练权重，100验证图片，2次信道重复，
AWGN/中FSO/强FSO × 0/5/10/20 dB，共4×100×2×3×4=9600次前向。
`--seed 42` 仅固定配对抽样，不新增训练或多训练种子实验。
可先 `--samples 4 --repeats 1 --profiles awgn fso_moderate --snr-list 0 10` 做64次联调。

## 运行

从已有 multimodel 根目录执行，沿用 Python3.8 / torch1.12，GPU物理1。
代码不安装依赖，不改全局设置，输出已存在即停止。

```bash
cd /data1/zw/WYH/multimodel
export CUDA_VISIBLE_DEVICES=1
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
"$PYTHON_BIN" -B v10_extension/pilot_gain/test_pilot_gain.py
"$PYTHON_BIN" -B v10_extension/pilot_gain/test_channel_pairing.py
"$PYTHON_BIN" -B v10_extension/pilot_gain/run.py \
  --checkpoint experiment_results/v10_formal_paper_v10_02_noamp/train_mixed_seed42/best.pth \
  --split-manifest v10_extension/manifests/ln_v10_splits.json \
  --output-dir experiment_results/v10_pilot_gain_v1 \
  --data-root COCO2017 \
  --vae-path pretrained/sd-vae-ft-mse \
  --text-model-path pretrained/clip-vit-base-patch32 \
  --max-total-hours 2
```

加入 `--dry-run` 只查看计划，不创建输出、不加载GPU。
建议先把 output-dir 改为 `experiment_results/v10_pilot_gain_smoke_v1`，加上前述64次参数运行；
成功后再用默认100图的全量诊断。总时限2小时包含4组；超时保留失败，不无限续跑。
加载本地骨干四次的成本、GPU共享负载和I/O决定实际时间，不能把时限当耗时预测。

进度：主入口显示 START/DONE；每组日志记录 images/总数。
另开终端 `tail -n 20 experiment_results/v10_pilot_gain_v1/*.log`。
无新行可能正在加载骨干，不能据此判定死锁；不停止其他用户进程。

## 配对与结果

套件检查完整图像×信道×SNR×repeat笛卡尔积、完成证据、同权重/划分/旧源码/设置、
相同导频和衰落哈希、载荷形状及载荷噪声生成器状态、相同传输资源。任何失配拒绝汇总。
新代码的SHA-256在plan中记录，原eval的运行身份、原始逐条样本和输入哈希保留。

`gain_summary.json`：绝对指标、三套差值和置信区间、模式占比、检索R@1。
`gain_table.csv`：可用Excel查看。`REPORT.md`：PSNR/BLEU2增益表。
差值=自适应−固定策略；PSNR/SSIM/BLEU等正值好，header_rejected负值好。
同图重复先平均，再按图片做配对bootstrap；95%区间是探索性逐项区间，
不代表多重检验校正后的显著性，也不包括训练随机性的方差。
Common检索来自每组图库级汇总，只提供点估计，不伪造逐图检索置信区间。
不删除失败或低质量样本、不事后挑最佳固定模式冒充预设基线。

发回整个新结果目录（含samples.jsonl/summary/运行身份/完成标记/日志/少量images），
不需要发送best.pth、数据集或预训练模型。主套件COMPLETE仅在配对验收和报告完成后生成。

## 本地与服务器验证区分

test_pilot_gain.py是纯标准库人工数据验收测试，不等于真实神经网络数值测试。
test_channel_pairing.py用实际torch信道检查4策略的配对和资源，无需CLIP/VAE或GPU；
本地没有torch时显示SKIP，不算数值通过。服务器应看到该项OK。
服务器小样本联调必须先通过，随后默认验证集实验才可支撑恢复增益结论。
