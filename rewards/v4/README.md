# Reward v4: Structural similarity reward

## Archive status

- 状态：`complete`
- 小型结果文件：1
- 结果目录：[`results/reward_v4`](../../results/reward_v4/)

## Motivation

使用Table、Column和Join结构相似度为错误SQL提供更密集的信号。

## Reward design

加入结构相似度、Final一致性、失败Final、未验证Final和失败SQL复用等分量。

规范入口为[`reward.py`](reward.py)。
入口链接到`../source/`中保留原始文件名的源码。

## Key result

五步验证Exec 38.00%→36.00%，SQLExec 67.00%→68.00%，ToolErr 32.50%→28.00%。

## Decision

正确性没有改善，结构奖励可能奖励语义错误SQL。
