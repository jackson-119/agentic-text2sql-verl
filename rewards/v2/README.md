# Reward v2: Initial process reward

## Archive status

- 状态：`complete`
- 小型结果文件：1
- 结果目录：[`results/reward_v2`](../../results/reward_v2/)

## Motivation

建立执行正确性、SQL可执行性、格式和工具过程之间的基础组合奖励。

## Reward design

联合奖励终局正确性、SQL执行和Agent工具流程，是目前保存下来的最早可执行Reward。

规范入口为[`reward.py`](reward.py)。
入口链接到`../source/`中保留原始文件名的源码。

## Key result

仓库保存离线复算结果，主要作为Reward v3的历史起点。

## Decision

保留为最早可执行历史版本。

## Comparison caveat

只有基础模型、SFT数据、训练预算、采样参数和评测集一致时，
不同Reward之间才能视为严格的同预算比较。
