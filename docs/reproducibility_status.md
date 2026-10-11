# 复现状态

本页区分组件迁移、历史实验、接口验证与新仓库端到端复现。
历史实验没有因整理仓库而重新运行。

| 内容 | 已有证据 | 尚未覆盖 |
|---|---|---|
| 标量奖励与 SQL 评估 | 单元测试；160 条历史训练轨迹评分一致 | 新仓库在线 Reward Manager |
| 数据库工具 | 后端测试和异步生命周期预检 | 完整 Agent 在线交互及安全隔离保证 |
| SFT 数据与监督 | 构建器、真实 Parquet 集成、3 条真实 Tokenizer 合成轨迹对齐 | 历史完整数据重建、regex 警告核验 |
| 单卡 SFT 入口 | Dataset/Collator、CLI 与输出保护测试 | 真实模型训练、历史启动参数完整复原 |
| GRPO 准备与启动 | 5+6 项测试、Hydra Compose、Dry-run、独立 CPU Ray worker 环境传递及评分测试 | GPU 训练与在线 Reward Manager |
| Token-level IS | 9 项测试和 64 组历史权重分支对照 | 实际采样概率语义的完整核验 |
| SC 核心与辅助 log-prob | PPO 系数、解析梯度、packed 因果对齐测试 | 新仓库实际 Actor 在线集成 |
| SC 框架补丁 | 指定本地基线快照的临时副本上应用，13 个目标文件哈希及语法核对 | 干净上游兼容性、运行时集成、完整许可审查 |
| 累计过程回报 | 核心逐字迁入，10 项 CPU 测试和历史结果登记 | 在线事件奖励映射及 Reward Manager 接线 |
| 实验结果 | Control/IS/SC 曲线与配对统计，累计回报历史比较 | 多训练种子和独立最终测试 |
| 环境与分发 | 可选依赖分组、来源记录 | 完整依赖锁定及干净环境端到端复现 |

## 当前测试证据

统一清单登记 14 个模块、119 项：
core 43、sft 18、grpo 11、algorithms 37、tools 10。

当前已在已有环境完成 14 个模块、119 项全量 CPU 回归，全部通过且无跳过。
记录见 [全量回归摘要](../results/testing/cpu_regression.json)；这不是干净环境或在线训练验证。
真实 Tokenizer 三轨迹检查和独立 Ray worker 检查是额外证据，不重复计入单元测试数量。

真实 Tokenizer 标准/恢复/验证轨迹为 705/773/705 个 token，均有 79 个监督 token。
恢复轨迹错误调用保留为上下文，不参与监督。加载时的 regex 警告尚未解决。

## 解释边界

- 历史训练来自原工程，不是当前仓库重新训练所得。
- 原始 V10 最优模型与后续 Control 重跑不是同一次实验。
- SC 与 Control 存在 Actor 编译设置差异。
- Pilot200 和反复使用的 Selection417 不作为独立最终测试。
- 历史 Holdout 使用情况按具体实验记录，不作整体从未使用的声明。
- 梯度机制验证不等于任务收益，也不等于 Adam 更新方向验证。
- Token 级截断 IS 不宣称完整无偏策略修正。
- 补丁验证针对指定本地快照，Git HEAD 不能代表其中所有本地改动。
- 来源哈希和测试通过均不替代版权与许可核对。

## 详细证据

- [SFT 迁移](../results/sft/migration_validation.json)
- [GRPO 入口](grpo_launcher.md)与[迁移摘要](../results/grpo/migration_validation.json)
- [累计过程回报](cumulative_credit.md)
- [SC 补丁](sc_patch_bundle.md)与[来源清单](../experiments/manifests/sc_patch_bundle.json)
- [测试清单](../tests/README.md)与[依赖边界](dependencies.md)
