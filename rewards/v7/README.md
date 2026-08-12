# Reward v7: Correctness-first reward

## Archive status

- 状态：`complete`
- 小型结果文件：1
- 结果目录：[`results/reward_v7`](../../results/reward_v7/)

## Motivation

修复Reward v6中错误SQL获得正奖励的问题。

## Reward design

所有错误轨迹保持负分，正确与错误轨迹严格分离。

规范入口为[`reward.py`](reward.py)。
入口链接到`../source/`中保留原始文件名的源码。

## Key result

离线复算中正分错误轨迹为0，奖励与正确性相关系数约为0.997。

## Decision

离线不变量通过，随后进一步简化为Reward v8。
