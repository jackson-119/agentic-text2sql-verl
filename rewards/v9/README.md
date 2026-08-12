# Reward v9: Verified process reward

## Archive status

- 状态：`complete`
- 小型结果文件：10
- 结果目录：[`results/reward_v9`](../../results/reward_v9/)

## Motivation

缓解Reward v8中大量全错误GRPO组没有信用分配信号的问题。

## Reward design

加入工具顺序、schema检查、协议、执行验证及过程惩罚。

规范入口为[`reward.py`](reward.py)。
入口链接到`../source/`中保留原始文件名的源码。

## Key result

Reward信息组率接近100%，但Selection417中v9 Step20为51.08%，低于v8的52.04%；Step32发生协议坍缩。

## Decision

拒绝；密集过程奖励产生了可被策略利用的代理目标。
