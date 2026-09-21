# V10 项目内运行版

2026-09-20。只在现有 `/data1/zw/WYH/multimodel` 内安装和运行。此版替代之前的并列目录部署方式，不再修改项目外配置，也不删除过去产生的项目外文件。

只需上传 `multimodel_v10_inplace_20260920.pyz` 到上述目录。它是带安全安装器的代码压缩包，直接用现有 Python 运行；不要用 pip，不要解压旧 tar.gz。

```bash
cd /data1/zw/WYH/multimodel
export PYTHON_BIN=/home/zw/anaconda3/envs/torch/bin/python
"$PYTHON_BIN" -B multimodel_v10_inplace_20260920.pyz --project-root . --dry-run
"$PYTHON_BIN" -B multimodel_v10_inplace_20260920.pyz --project-root .
bash run_v10.sh package
bash run_v10.sh check
```

任一步报错就停在该步，发回错误，不跳过验收、不覆盖冲突文件。安装器先检查全部冲突才写文件；重复安装相同内容不会改写它。

新增结构：

```text
multimodel/
├── 原来的代码、数据、权重、结果       保持原样
├── run_v10.sh                      新版统一入口
├── README_V10_INPLACE.md
├── v10_extension/                 新版模型副本、信道、脚本、配置、测试
│   ├── manifests/                 固定数据划分，首次运行生成
│   └── verification/              检查报告
├── experiment_results/v10_*        新版实验，要求新目录
├── logs/v10/                      新版运行日志
└── .cache/v10/                    新版临时文件及库缓存
```

完整的划分、初始化权重确认、预检、训练、AWGN/Rayleigh/FSO 评估、Common-only 和扩散预览命令，都在 [运行说明](v10_extension/README_V10.md)。物理假设与研究限制见 [设计](v10_extension/docs/DESIGN.md)，当前测试边界见 [交接](v10_extension/docs/HANDOFF.md)。

本轮改变安装和路径约束，保留先前 V10-A 算法。部署检查不代表效果已提升；正式训练前必须在服务器现有环境通过完整检查与小规模预检。
