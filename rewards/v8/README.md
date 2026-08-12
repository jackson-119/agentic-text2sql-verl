# Reward v8: Binary execution reward

## Archive status

- 状态：`complete`
- 小型结果文件：10
- 结果目录：[`results/reward_v8`](../../results/reward_v8/)

## Motivation

只依据最终SQL执行正确性进行训练，避免复杂代理指标误导策略。

## Reward design

执行正确奖励1，执行错误奖励0，不直接奖励结构、工具外观或过程。

规范入口为[`reward.py`](reward.py)。
入口链接到`../source/`中保留原始文件名的源码。

## Key result

1.5B Final Holdout中SFT为47.48%，v8 Step32为55.40%，提升7.91个百分点，McNemar p=0.00000870。

## Decision

接受并作为锁定主基线；选择Step32以避开后期策略坍缩。
