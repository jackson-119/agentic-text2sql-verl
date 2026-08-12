# Reward v5: Verified structural reward

## Archive status

- 状态：`complete`
- 小型结果文件：6
- 结果目录：[`results/reward_v5`](../../results/reward_v5/)

## Motivation

阻止失败、未执行和未验证SQL获得结构奖励。

## Reward design

只有通过执行验证门控的SQL才能获得Table、Column和Join等结构奖励。

规范入口为[`reward.py`](reward.py)。
入口链接到`../source/`中保留原始文件名的源码。

## Key result

Final Holdout中v3 Step20到v5 Step48：Exec 36.69%→40.05%，SQLExec 68.82%→86.09%，ToolErr 30.22%→14.87%。

## Decision

接受；显著改善可执行性和工具稳定性。
