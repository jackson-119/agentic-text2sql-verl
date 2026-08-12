# Reward v6: Semantic-first structural reward

## Archive status

- 状态：`complete`
- 小型结果文件：9
- 结果目录：[`results/reward_v6`](../../results/reward_v6/)

## Motivation

进一步优化Table选择、Join结构和SQL语义。

## Reward design

提高语义结构奖励和semantic gap分量，区分不同程度的错误SQL。

规范入口为[`reward.py`](reward.py)。
入口链接到`../source/`中保留原始文件名的源码。

## Key result

Selection417中v5为38.61%，v6 Step24为36.45%；语义奖励与正确性的相关系数为-0.648。

## Decision

拒绝；代理奖励与最终正确性方向错位。
