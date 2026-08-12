# Agentic Text-to-SQL Reward Experiments

本仓库整理基于VERL、Qwen2.5-Coder和Spider开展的
Agentic Text-to-SQL SFT与GRPO奖励函数实验。

仓库保存Reward v1至v10的实现、紧凑结果，以及
Reward hacking、协议坍缩和语义策略坍缩等失败经验。

## Repository layout

```text
rewards/                     Reward源码和版本说明
results/                     小型JSON/CSV结果
comparisons/                 对比与配对检验索引
docs/                        Reward演进和结论
```

## Reward evolution

| Version | Main idea | Decision | Archive |
|---|---|---|---|
| v1 | Early process reward prototype | 归档为results-only，不使用推测代码冒充真实Reward。 | `results_only` |
| v2 | Initial process reward | 保留为最早可执行历史版本。 | `complete` |
| v3 | Process, recovery and duplication control | 流程塑形有效，但SQL语义错误仍是主要瓶颈。 | `complete` |
| v4 | Structural similarity reward | 正确性没有改善，结构奖励可能奖励语义错误SQL。 | `complete` |
| v5 | Verified structural reward | 接受；显著改善可执行性和工具稳定性。 | `complete` |
| v6 | Semantic-first structural reward | 拒绝；代理奖励与最终正确性方向错位。 | `complete` |
| v7 | Correctness-first reward | 离线不变量通过，随后进一步简化为Reward v8。 | `complete` |
| v8 | Binary execution reward | 接受并作为锁定主基线；选择Step32以避开后期策略坍缩。 | `complete` |
| v9 | Verified process reward | 拒绝；密集过程奖励产生了可被策略利用的代理目标。 | `complete` |
| v10 | Correctness-gated process reward | 选择扩展SFT Reward v10 Step88；门控避免v9式协议坍缩，但长期训练仍出现语义坍缩。 | `complete` |

详细说明见[Reward evolution](docs/reward_evolution.md)。

## Main results

### Reward v8

- SFT baseline：47.48%
- Reward v8 Step32：55.40%
- 绝对提升：+7.91个百分点
- Bootstrap 95% CI：[+4.56%, +11.27%]
- McNemar p：0.00000870

### Expanded SFT with Reward v10

- 未扩展v10 Step88：48.20%
- 扩展SFT v10 Step88：54.20%
- 绝对提升：+6.00个百分点
- McNemar p：0.00155

## Repository scope

包含Reward源码、指标扩展、小型结果、配对检验和模型选择记录。

不包含模型权重、Checkpoint、原始数据、完整rollout、
generated输出、W&B、Ray或训练日志。

## Upstream

VERL commit：`e003163181731412595257a72ec173071efb125f`

## License

见[LICENSE](LICENSE)。
