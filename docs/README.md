# 文档导航

## 环境、数据与评测

- [数据库 Agent 工具](agent_environment.md)
- [SFT 数据与监督掩码](sft_pipeline.md)
- [SFT 数据构建](sft_data_construction.md)
- [单卡 SFT 运行模板](../configs/sft/README.md)
- [依赖分组与环境记录](dependencies.md)
- [SQL 执行评估语义](evaluation_semantics.md)
- [正确性门控奖励](reward_design.md)
- [标量奖励入口](reward_integration.md)

## 策略修正

- [Token-level IS](importance_sampling.md)
- [SC 修正核心](score_centering.md)
- [SC 辅助 log-prob 与序列对齐](sc_auxiliary.md)
- [Control / IS / SC 实验结果](policy_correction_results.md)

## 工程与复现

- [基础配置模板](baseline_configuration.md)
- [GRPO 准备与启动入口](grpo_launcher.md)
- [SC 历史框架补丁](sc_patch_bundle.md)
- [测试入口与依赖](testing.md)
- [复现状态与限制](reproducibility_status.md)

SFT 构建与单卡入口、GRPO 准备与启动入口、累计回报核心和 SC 历史补丁均已整理。
组件迁移及临时补丁验证不等于新仓库已完成端到端训练复现。

## 累计过程回报组件

- [计算方式、输入契约与历史结果](cumulative_credit.md)
- [历史结果摘要](../results/cumulative_credit/historical_summary.json)
- [来源与迁移测试记录](../experiments/manifests/cumulative_credit_source.json)
