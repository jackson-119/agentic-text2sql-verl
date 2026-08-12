# Key Findings

## SFT协议正确性是RL的前提

修复1.5B SFT loss mask后，Pilot200执行准确率达到42.50%，
工具调用率达到100%。

## 简单Reward非常有竞争力

Reward v8在锁定Final Holdout上将执行准确率从47.48%
提高到55.40%。

## 密集代理奖励可能产生Reward hacking

Reward v6语义奖励与正确性负相关；Reward v9提高信息组率，
但最终发生协议坍缩。

## 正确性门控只能降低风险

Reward v10避免了v9式协议坍缩，但长期训练仍可能发生
SQL语义策略坍缩。

## 扩展SFT收益主要在RL阶段体现

扩展SFT配合相同Reward v10和Step88预算后，Selection417
执行准确率提高6.00个百分点。

## 最终Checkpoint通常不是最佳Checkpoint

必须通过独立Selection集选择Checkpoint，不能默认最后一步最佳。
