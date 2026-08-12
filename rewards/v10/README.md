# Reward v10: Correctness-gated process reward

## Archive status

- 状态：`complete`
- 小型结果文件：21
- 结果目录：[`results/reward_v10`](../../results/reward_v10/)

## Motivation

保留过程排序能力，同时禁止错误轨迹通过良好过程获得更高奖励。

## Reward design

错误轨迹固定为-0.1且过程bonus为0；只有执行正确轨迹才能获得过程bonus。

规范入口为[`reward.py`](reward.py)。
入口链接到`../source/`中保留原始文件名的源码。

## Key result

扩展SFT同预算Step88相对未扩展Step88从48.20%提高到54.20%，提升6.00个百分点，p=0.00155。

## Decision

选择扩展SFT Reward v10 Step88；门控避免v9式协议坍缩，但长期训练仍出现语义坍缩。

## Comparison caveat

只有基础模型、SFT数据、训练预算、采样参数和评测集一致时，
不同Reward之间才能视为严格的同预算比较。
