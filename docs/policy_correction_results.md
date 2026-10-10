# 采样策略修正：Control / IS / SC

## 研究问题

在保持 correctness-gated 标量奖励、不使用累计过程回报的情况下，
比较 token-level IS 和前缀条件 Score Centering 是否改善执行准确率。

Control 是本次对照重跑，不是历史最优原始 V10 Step88 模型。
IS 使用上限为 2 的 token 级截断权重，不宣称实现完整无偏策略修正。
SC 使用每个前缀的 4 个辅助样本，并结合实际 PPO 梯度系数处理裁剪。
具体实现与训练补丁仍需完成迁入；本文先登记历史结果。

## 结果

Pilot200 是反复使用的开发集，不是独立最终测试集。
主要比较固定 Step120，最佳 checkpoint 仅作为探索性结果。

| 方法 | Step0 | Step120 | 增益 | 开发集最佳（探索性） |
|---|---:|---:|---:|---|
| CONTROL | 43.50% | 52.50% | +9.00 pp | Step48：53.50% |
| IS | 43.00% | 50.00% | +7.00 pp | Step56：51.50% |
| SC | 44.00% | 52.50% | +8.50 pp | Step88：53.00% |

固定 Step120：

- IS − Control：−2.50 pp，逐题配对 bootstrap 95% CI [−7.00, +1.50] pp；
  McNemar exact p=0.3593。
- SC − Control：0.00 pp，95% CI [−3.50, +3.50] pp；
  McNemar exact p=1.0000。
- SC 与 Control 各有 6 道独占正确题；准确率相同不代表预测相同。

本轮未显示 IS 或 SC 相对 Control 的明确执行准确率收益。
不能将“不显著”解释为方法等效，也不能据此断言方法普遍无效。

## 解释限制

- 每组只有一次训练，逐题置信区间不覆盖训练种子的不确定性。
- 最佳 checkpoint 在同一开发集选择与报告，存在选择偏差。
- SC 关闭 Actor 编译，Control / IS 开启，存在工程配置差异。
- 三组 Step0 结果并不完全一致；配置一致不保证采样与执行逐位一致。
- SC Step13–120 的完整运行指标核验仍有缺口。
- 配对 p 值未经多重比较校正；本报告是描述性开发集分析。
- 使用项目自定义 SQL 执行比较语义，不等同于官方 Spider 指标。
- 结果来自历史训练环境；新仓库尚未完成端到端训练复现。
- 历史其他实验的 Holdout 使用情况应单独登记，不能整体宣称从未使用。

## 可追溯证据

- [结构化汇总](../results/policy_correction/summary.json)
- [来源及输入哈希](../experiments/manifests/policy_correction_source.json)
- [SQL 评估语义](evaluation_semantics.md)

本次重新核对了曲线算术、逐题配对计数和 McNemar p 值；
bootstrap 区间直接保留原报告，未重新抽样计算。
未公开逐题 prompt、SQL、数据库或模型权重。
