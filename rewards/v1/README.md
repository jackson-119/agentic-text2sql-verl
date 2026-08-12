# Reward v1: Early process reward prototype

## Archive status

- 状态：`results_only`
- 小型结果文件：3
- 结果目录：[`results/reward_v1`](../../results/reward_v1/)

## Motivation

验证Agent工具轨迹能否通过自定义奖励参与GRPO训练。

## Reward design

原始源码没有进入Git历史、当前源码目录或备份；只能确认实验完成20个训练Step，并保存了Step 0、10、20验证结果。

原始源码未恢复。
本目录故意不提供`reward.py`，避免用猜测代码冒充真实实现。

## Key result

Reward v1训练结果及v1与v3的离线比较仍然存在，但无法可靠恢复原始奖励公式。

## Decision

归档为results-only，不使用推测代码冒充真实Reward。

## Comparison caveat

只有基础模型、SFT数据、训练预算、采样参数和评测集一致时，
不同Reward之间才能视为严格的同预算比较。
