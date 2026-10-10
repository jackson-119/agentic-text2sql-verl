# Agentic Text2SQL with verl

面向多轮数据库工具调用的语言模型后训练项目，围绕
**SFT 监督构造、GRPO 奖励、信用分配与采样策略修正**开展实验。

本仓库整理可独立测试的组件、可追溯的历史结果和方法说明。
当前不是完整训练发行版；历史训练由原实验工程完成，
尚未在本仓库完成端到端复现。

## 已实现与验证

- **数据库工具**：列出表、查询 schema、执行只读 SQL。
- **SFT 监督**：Assistant-only 标签构造、生成头排除、前缀一致性检查。
- **奖励与评估**：正确性门控标量奖励、SQL 提取与执行结果比较。
- **策略修正组件**：token-level 截断 IS、实际 PPO 系数驱动的 SC、
  packed 序列辅助 log-prob 计算。
- **组件测试**：8 个模块、80 项测试在已有环境通过。
- **迁移证据**：160 条历史训练轨迹的标量评分回放一致。

IS 参考实现基于 verl 现有权重分支，不作为新提出的算法。
SQL 指标使用本项目的执行比较语义，不冒称官方 Spider 评测。

## 历史实验结果

Pilot200 为反复使用的开发集。以下比较固定 Step120：

| 方法 | 执行正确数 | 执行准确率 |
|---|---:|---:|
| Control：标量奖励 + GRPO | 105/200 | 52.50% |
| Token-level IS，上限 2 | 100/200 | 50.00% |
| Score Centering，4 个辅助样本 | 105/200 | 52.50% |

本轮没有显示 IS 或 SC 相对 Control 的明确准确率收益；
“不显著”不等于方法等效。每组只有一次训练，
SC 与 Control 还存在 Actor 编译配置差异。

Control 是本次对照重跑，不是历史最优原始 V10 模型。

详见 [结果与统计限制](docs/policy_correction_results.md)、
[SFT 数据与监督方式](docs/sft_pipeline.md)。

## 运行组件测试

在仓库根目录执行。下面三组依赖不同，不会自动安装软件：

    python scripts/run_tests.py --group core
    python scripts/run_tests.py --group algorithms
    python scripts/run_tests.py --group tools

core 覆盖奖励、SQL 评估和 SFT 掩码；
algorithms 需要 PyTorch；tools 需要已有 verl 工具接口依赖。
完整运行使用 --group all。

这不是训练命令，也不是干净环境安装验证。
详见 [测试与依赖](docs/testing.md)。

## 仓库导航

| 目录 | 当前内容 |
|---|---|
| src/agentic_text2sql/ | 工具、SFT 掩码、奖励评估、IS/SC 独立组件 |
| tests/ | CPU 单元测试和接口回归 |
| scripts/ | 统一测试入口 |
| configs/ | GRPO smoke 配置与数据库工具模板 |
| experiments/ | 实验登记、源码和结果来源哈希 |
| results/ | 精简历史证据与比较结果，不含权重或原始数据 |
| docs/ | 方法、测试、结果和复现边界 |
| patches/ | 框架补丁预留位置，尚未发布可安装补丁 |

## 尚未完成

- 数据构建与完整 SFT/GRPO 训练入口迁入。
- 累计过程回报组件及相关结果整理。
- vLLM 辅助采样、Agent 传输、Actor 接线的完整适配。
- 干净环境依赖锁定、端到端训练复现。
- 上游来源、版权声明与许可材料的完整核对。

Reward V1–V10 不再按版本组织目录；
奖励设计作为整体后训练研究的一部分保留。

## 证据边界与来源

Pilot200 和反复使用的 Selection417 不作为独立最终测试。
历史 Holdout 使用情况按实验记录，不作整体“从未使用”的声明。
梯度机制验证不等于任务效果提升，参数梯度也不等于优化器更新。

参见 [复现状态](docs/reproducibility_status.md)、
[完整文档导航](docs/README.md)及
[实验登记](experiments/registry.json)。

保留现有 LICENSE；来源哈希不替代版权和许可核对。
本仓库不提供模型权重、Spider 数据库或完整私有运行日志。
