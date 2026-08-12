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

| Version | Design | Analysis |
| --- | --- | --- |
| v1 | 终局结果加权奖励：执行正确性占0.90，SQL可执行性占0.05，严格输出格式占0.05。 | 本项目首个在线Agentic GRPO奖励。Pilot200五步训练后执行准确率从25.00%升至34.50%，但奖励信号较稀疏，重复执行率和工具错误率仍分别为35.50%和37.50%。 |
| v2 | 在终局奖励之外，增加列出数据表、查询Schema、Schema后执行SQL、错误恢复和有序工具链等奖励。 | 仅完成离线原型验证。对v1的160条轨迹复算后，信息组率达到27.50%，但没有进行独立的在线GRPO训练，也没有生成独立Checkpoint。 |
| v3 | 在过程奖励基础上加入工具调用效率奖励，以及重复调用、畸形调用和遇错后停止等惩罚。 | 在线训练至Step20后，Pilot200执行准确率达到38.00%，最终答案率和格式率达到100%，重复execute率降至0%，形成首个稳定的过程感知模型。 |
| v4 | 加入SQL结构相似度、错误恢复、最终SQL一致性、失败SQL复用、未验证最终SQL和失败最终SQL等项。 | 工具错误有所下降，但Pilot200执行准确率从38.00%降至Step10的34.50%。结构代理奖励没有稳定转化为语义正确性。 |
| v5 | 只允许经过执行验证的SQL获得结构奖励，失败或未验证的SQL不能获得表、列和JOIN结构奖励。 | 最终选择Step48。Final Holdout执行准确率由v3的36.69%提升至40.05%，绝对提升3.36个百分点；工具错误率由30.22%降至14.87%。 |
| v6 | 采用语义优先设计，以SQL结构相似度提供正奖励，并通过语义差距惩罚区分可执行但错误的SQL。 | 该设计被拒绝。语义代理奖励与执行正确性的相关系数为-0.648，Selection417执行准确率由v5的38.61%下降至36.45%。 |
| v7 | 建立正确性优先的奖励层级：正确轨迹获得正分，所有错误轨迹保持负分，避免错误SQL取得正向代理奖励。 | 离线不变量全部通过，奖励与正确性的相关系数约为0.997，且不存在正分错误轨迹；但全错误组仍较多，未选出独立的在线训练模型。 |
| v8 | 使用最简单的二值执行奖励，只判断预测SQL执行结果是否满足目标结果，不奖励中间工具过程。 | 配合协议修复后的1.5B模型，Step32在Selection417上由SFT的44.36%提升至52.04%，在Final Holdout上由47.48%提升至55.40%；继续训练则出现策略坍缩。 |
| v9 | 在终局正确性之外加入经过验证的过程奖励和过程惩罚，以提升GRPO组内奖励差异和过程信用分配能力。 | Reward信息组率达到100%，但Step20在Selection417上的执行准确率为51.08%，没有超过v8的52.04%；Step32还出现协议坍缩，因此未被采用。 |
| v10 | 采用正确性门控过程奖励：所有错误轨迹统一得到负分，只有执行正确的轨迹才能获得过程奖励。 | 消除了错误SQL利用过程奖励得分的问题，并显著延缓坍缩。扩展SFT后的Step88在Selection417达到54.20%，比v8高2.16个百分点；但Step128仍出现语义策略坍缩。 |

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

Reward v1用于验证自定义奖励能否接入Agent GRPO训练。

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
