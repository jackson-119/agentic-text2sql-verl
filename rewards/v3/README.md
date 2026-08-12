# Reward v3: Process, recovery and duplication control

## Archive status

- 状态：`complete`
- 小型结果文件：6
- 结果目录：[`results/reward_v3`](../../results/reward_v3/)

## Motivation

解决最终答案缺失、重复execute、无效调用和错误恢复不足。

## Reward design

奖励正确终局、可执行SQL、工具顺序和错误恢复，并惩罚重复execute、重复schema和冗余调用。

规范入口为[`reward.py`](reward.py)。
入口链接到`../source/`中保留原始文件名的源码。

## Key result

Pilot200 Step 0到20：Exec 25.50%→38.00%，SQLExec 39.00%→67.50%，Final 39.50%→100%，重复execute 72.50%→0%。

## Decision

流程塑形有效，但SQL语义错误仍是主要瓶颈。
