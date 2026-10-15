# 图文多模态语义通信项目

研究在受限且不可靠信道中传输图像与文本，探索Common共享语义、连续JSCC与接收端恢复。
历史成果与当前实测需分开阅读，源码实现不等于模型有效。

## 接续入口（2026-10-15）

- [当前状态](v10_extension/docs/CURRENT_STATUS.md)
- [V12已核验实测结果与图表](v10_extension/docs/results/v12_semantic_probe_20261015/README.md)
- [V13离线语义评价：协议、代码、运行步骤](v10_extension/docci_semantic_eval_v13/README.md)
- [V11历史文本续训结果](v10_extension/docs/results/v11_text_continue_20261014/README.md)
- [共享交接记录](v10_extension/docs/HANDOFF.md)
- [智能体工程规则](AGENT_ENGINEERING_GUIDELINES.md)
- [项目详细介绍与版本演进](docs/项目详细介绍与版本演进.md)

V12同100条DOCCI qual_dev的clean宏平均规范化WER：CE 45.27%、CE+KD 44.75%；
KD额外收益区间跨零，两组仍QUALITY_STOP、全文一致0/100。
图像接收适配在人工mask25下有约0.54dB收益，但正确/打乱oracle文本基本等价，未证明图文互促。
V13新增离线评价工作流，人工事实审核与真实BERTScore推理尚未完成；没有新训练、FSO或同资源数字胜出结论。

公开内容以源码、协议、合成测试和派生汇总为主，不上传数据集原始图文、预训练/实验权重或完整逐条records。
暂未为自有代码新指定项目级开源许可证；第三方许可证不变，仓库公开不代表这些资产被重新授权。
