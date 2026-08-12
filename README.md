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

| Version | Description |
|---|---|
| v1 | 早期过程奖励原型，用于验证自定义Reward能否驱动Agent GRPO训练；实验结果存在，但源码未归档，无法复核具体公式。 |
| v2 | 联合执行正确性、SQL可执行性、格式和工具过程进行奖励；多个目标耦合较粗，难以判断模型具体优化了哪一项。 |
| v3 | 通过流程塑形、错误恢复和重复调用惩罚改善Agent行为；显著提高Final率并消除重复execute，但SQL语义错误仍是瓶颈。 |
| v4 | 使用Table、Column和Join结构相似度提供密集信号；结构接近不等于语义正确，错误SQL可能获得不合理的正反馈。 |
| v5 | 只允许经过执行验证门控的SQL获得结构奖励；明显改善可执行率和工具稳定性，但可执行且语义错误的SQL仍较多。 |
| v6 | 进一步强化Table、Join和语义结构奖励；代理奖励发生目标错位，语义奖励与最终正确性呈负相关。 |
| v7 | 恢复正确性优先原则，严格分离正确与错误轨迹；消除了正分错误轨迹，但正确性信号稀疏，全错误组仍缺少梯度。 |
| v8 | 采用纯二值执行正确性奖励，使训练目标直接对齐最终指标；取得最可靠的提升，但全零GRPO组较多，后期仍会发生策略坍缩。 |
| v9 | 加入工具顺序、协议和验证过程奖励以提高信用分配密度；过程代理目标可被策略利用，训练后期发生严重协议坍缩。 |
| v10 | 仅允许执行正确轨迹获得过程bonus，错误轨迹固定为负分；避免了v9式协议坍缩，但长期训练仍可能发生SQL语义策略坍缩。 |

详细设计、实验数据和版本结论见[Reward evolution](docs/reward_evolution.md)。

## Results

不同Reward版本使用过不同模型、训练预算和评测集。
以下结果按版本分别展示；只有明确标注的配对实验可以作严格比较。

### Reward v1: Early process reward prototype

| Metric | Value |
|---|---:|
| Evaluation | 早期Pilot20实验 |
| Training steps | Step 1–20 |
| Validation checkpoints | Step 0、10、20 |
| Source status | 原始Reward源码未恢复 |
| Result status | `results_only` |

**Analysis**

Reward v1用于验证自定义奖励能否接入Agent GRPO训练。 现存日志、rollout路径、Checkpoint和验证文件证明实验真实运行过， 但原始奖励函数没有进入Git历史、当前源码目录或备份。 因此本仓库只归档结果，不根据后续版本猜测其奖励公式。

### Reward v2: Initial process reward

| Metric | Value |
|---|---:|
| Evaluation | 既有轨迹离线复算 |
| Online paired result | 未形成可比的正式在线结果 |
| Source status | 完整 |
| Historical role | 最早保存下来的可执行Reward |

**Analysis**

Reward v2联合终局正确性、SQL可执行性、格式和工具过程进行计分。 它建立了后续过程奖励的基本结构，但多个目标耦合较粗， 难以判断模型究竟在优化最终正确性还是过程外观。

### Reward v3: Process, recovery and duplication control

| Metric | Value |
|---|---:|
| Evaluation | 0.5B Spider Pilot200，Step 0→20 |
| Execution accuracy | 25.50% → 38.00% |
| Absolute change | +12.50 pp |
| SQL executable | 39.00% → 67.50% |
| Final answer | 39.50% → 100.00% |
| Duplicate execute | 72.50% → 0.00% |
| Tool error | 47.50% → 32.00% |

**Analysis**

Reward v3显著改善了Agent流程行为：模型更稳定地输出Final SQL， 减少重复execute，并提高SQL可执行率。 这说明早期模型的主要问题不仅是SQL语义，还包括终止、格式、 工具调用顺序和错误恢复。 训练后剩余主要瓶颈转变为可执行但语义错误的SQL。

### Reward v4: Structural similarity reward

| Metric | Value |
|---|---:|
| Evaluation | 0.5B Spider Pilot200，Step 0→10 |
| Execution accuracy | 38.00% → 34.50% |
| Absolute change | -3.50 pp |
| SQL executable | 67.00% → 68.50% |
| Final answer | 100.00% → 99.50% |
| Tool error | 32.50% → 24.50% |

**Analysis**

Reward v4加入Table、Column和Join结构相似度，希望为错误SQL提供 比二值正确性更密集的学习信号。 虽然工具错误率下降、SQL可执行率略升，但执行准确率下降。 这证明结构接近并不等于语义正确，模型可能通过生成外观相似的 错误SQL获得不合理奖励。

### Reward v5: Verified structural reward

| Metric | Value |
|---|---:|
| Evaluation | 0.5B Final Holdout 417，v3 Step20→v5 Step48 |
| Execution accuracy | 36.69% → 40.05% |
| Absolute change | +3.36 pp |
| Bootstrap 95% CI | [+0.24 pp, +6.47 pp] |
| McNemar exact p | 0.0541 |
| SQL executable | 68.82% → 86.09% |
| FinalOK | 66.67% → 83.69% |
| Tool error | 30.22% → 14.87% |

**Analysis**

Reward v5要求SQL通过执行验证门控后才能获得结构奖励。 它显著提高SQL可执行率、Final一致性和工具稳定性，并在Final Holdout上得到正向准确率变化。 不过可执行但错误的SQL仍占较高比例，说明执行验证门控只能防止 失败SQL获奖，不能保证可执行SQL的语义正确。

### Reward v6: Semantic-first structural reward

| Metric | Value |
|---|---:|
| Evaluation | 0.5B Selection417，v5→v6 Step24 |
| Execution accuracy | 38.61% → 36.45% |
| Absolute change | -2.16 pp |
| Bootstrap 95% CI | [-4.08 pp, -0.48 pp] |
| McNemar exact p | 0.0352 |
| SQL executable | 81.77% → 88.49% |
| Executable but wrong | 43.17% → 52.04% |
| Semantic reward correlation | -0.648 |

**Analysis**

Reward v6进一步强化Table、Join和语义结构奖励，但形成了明确的 代理目标错位。 SQL可执行率提高的同时，执行准确率显著下降；217条可执行错误SQL 中有191条获得正语义奖励。 语义奖励与正确性呈负相关，因此Reward v6被正式拒绝。

### Reward v7: Correctness-first reward

| Metric | Value |
|---|---:|
| Evaluation | Selection417及既有训练轨迹离线复算 |
| Positive incorrect trajectories | 0 |
| Correct minimum reward | 约0.98–1.00 |
| Incorrect maximum reward | -0.10 |
| Reward/correctness correlation | 约0.997 |
| Reward-informative groups | 最高97.07% |
| Online selection result | 未形成完整正式模型选择结果 |

**Analysis**

Reward v7重新确立正确性优先原则，禁止错误SQL获得正奖励。 离线不变量和奖励相关性均表现良好，修复了Reward v6的方向性问题。 但它没有解决二值正确性本身的稀疏性：大量全错误组仍然缺少 能够直接提升正确率的有效梯度。

### Reward v8: Binary execution reward

| Metric | Value |
|---|---:|
| Evaluation | 1.5B Final Holdout 417，SFT→GRPO Step32 |
| Execution accuracy | 47.48% → 55.40% |
| Absolute change | +7.91 pp |
| Bootstrap 95% CI | [+4.56 pp, +11.27 pp] |
| McNemar exact p | 0.00000870 |
| SQL executable | 69.30% → 82.49% |
| Final answer | 69.30% → 85.61% |
| Tool error | 19.66% → 17.75% |

**Analysis**

Reward v8只依据最终SQL执行是否正确进行二值奖励，训练目标与 最终评测指标直接一致。 它取得了目前最强、最可靠的锁定Final Holdout提升，说明在工具 协议已经通过SFT稳定掌握后，简单正确性奖励可以优于复杂Reward。 缺点是全零GRPO组较多，训练后期有效梯度减少并发生策略坍缩， 因此最终选择Step32而不是最后一个Checkpoint。

### Reward v9: Verified process reward

| Metric | Value |
|---|---:|
| Evaluation | 1.5B Selection417，v8 Step32→v9 Step20 |
| Execution accuracy | 52.04% → 51.08% |
| Absolute change | -0.96 pp |
| McNemar exact p | 0.6516 |
| SQL executable | 81.06% → 82.49% |
| Final answer | 85.85% → 97.12% |
| Reward-informative groups | 接近100% |
| Step32 execution accuracy | 1.50% |
| Step32 protocol rate | 0.00% |

**Analysis**

Reward v9加入工具顺序、协议、schema检查和执行验证等奖励， 成功提高了GRPO组内的信用分配密度。 但过程奖励创造了可被模型利用的代理目标：模型能够优化过程分， 却不一定提高最终SQL正确性。 继续训练到Step32后发生严重协议坍缩，因此Reward v9被拒绝。

### Reward v10: Correctness-gated process reward

| Metric | Value |
|---|---:|
| Evaluation | 1.5B Selection417，未扩展→扩展SFT，同为Step88 |
| Execution accuracy | 48.20% → 54.20% |
| Absolute change | +6.00 pp |
| McNemar exact p | 0.00155 |
| Expanded Step88 SQL executable | 82.01% |
| Expanded Step88 Final answer | 98.56% |
| Expanded Step88 Tool error | 20.86% |
| Expanded Step112 execution | 54.44% |
| Expanded Step128 execution | 训练后期发生明显下降 |

**Analysis**

Reward v10将过程奖励限制在执行正确轨迹内：所有错误轨迹固定 为负分，错误轨迹的过程bonus恒为0。 这一门控避免了Reward v9式协议坍缩。扩展SFT与Reward v10结合后， 在相同Step88预算下比未扩展模型提高6.00个百分点。 Step112虽然点估计略高，但与Step88基本持平；根据预定义的早期 Checkpoint选择规则，最终选择Step88。 Step128出现SQL语义策略坍缩，但工具调用率仍为100%、畸形率为0， 说明这是语义策略退化，而不是协议失效。

### Overall conclusion

Reward演进表明，增加奖励分量并不必然改善最终正确性。
Reward v6和v9分别展示了语义代理奖励错位与过程代理目标利用。
Reward v8提供了最可靠的最终盲测证据；Reward v10则证明，
通过正确性门控可以更安全地加入过程bonus。

Reward v10 Step88目前仍属于Selection417上的后续候选，
不能替代已经完成独立Final Holdout盲测的Reward v8 Step32结论。

## Repository scope

包含Reward源码、指标扩展、小型结果、配对检验和模型选择记录。

不包含模型权重、Checkpoint、原始数据、完整rollout、
generated输出、W&B、Ray或训练日志。

## Upstream

VERL commit：`e003163181731412595257a72ec173071efb125f`

## License

见[LICENSE](LICENSE)。
