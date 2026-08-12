# Reward Evolution

项目从过程塑形出发，逐渐发现结构和过程代理奖励风险，
随后回到执行正确性，并对过程奖励施加正确性门控。

## Reward v1: Early process reward prototype

**Motivation:** 验证Agent工具轨迹能否通过自定义奖励参与GRPO训练。

**Design:** 原始源码没有进入Git历史、当前源码目录或备份；只能确认实验完成20个训练Step，并保存了Step 0、10、20验证结果。

**Result:** Reward v1训练结果及v1与v3的离线比较仍然存在，但无法可靠恢复原始奖励公式。

**Decision:** 归档为results-only，不使用推测代码冒充真实Reward。

## Reward v2: Initial process reward

**Motivation:** 建立执行正确性、SQL可执行性、格式和工具过程之间的基础组合奖励。

**Design:** 联合奖励终局正确性、SQL执行和Agent工具流程，是目前保存下来的最早可执行Reward。

**Result:** 仓库保存离线复算结果，主要作为Reward v3的历史起点。

**Decision:** 保留为最早可执行历史版本。

## Reward v3: Process, recovery and duplication control

**Motivation:** 解决最终答案缺失、重复execute、无效调用和错误恢复不足。

**Design:** 奖励正确终局、可执行SQL、工具顺序和错误恢复，并惩罚重复execute、重复schema和冗余调用。

**Result:** Pilot200 Step 0到20：Exec 25.50%→38.00%，SQLExec 39.00%→67.50%，Final 39.50%→100%，重复execute 72.50%→0%。

**Decision:** 流程塑形有效，但SQL语义错误仍是主要瓶颈。

## Reward v4: Structural similarity reward

**Motivation:** 使用Table、Column和Join结构相似度为错误SQL提供更密集的信号。

**Design:** 加入结构相似度、Final一致性、失败Final、未验证Final和失败SQL复用等分量。

**Result:** 五步验证Exec 38.00%→36.00%，SQLExec 67.00%→68.00%，ToolErr 32.50%→28.00%。

**Decision:** 正确性没有改善，结构奖励可能奖励语义错误SQL。

## Reward v5: Verified structural reward

**Motivation:** 阻止失败、未执行和未验证SQL获得结构奖励。

**Design:** 只有通过执行验证门控的SQL才能获得Table、Column和Join等结构奖励。

**Result:** Final Holdout中v3 Step20到v5 Step48：Exec 36.69%→40.05%，SQLExec 68.82%→86.09%，ToolErr 30.22%→14.87%。

**Decision:** 接受；显著改善可执行性和工具稳定性。

## Reward v6: Semantic-first structural reward

**Motivation:** 进一步优化Table选择、Join结构和SQL语义。

**Design:** 提高语义结构奖励和semantic gap分量，区分不同程度的错误SQL。

**Result:** Selection417中v5为38.61%，v6 Step24为36.45%；语义奖励与正确性的相关系数为-0.648。

**Decision:** 拒绝；代理奖励与最终正确性方向错位。

## Reward v7: Correctness-first reward

**Motivation:** 修复Reward v6中错误SQL获得正奖励的问题。

**Design:** 所有错误轨迹保持负分，正确与错误轨迹严格分离。

**Result:** 离线复算中正分错误轨迹为0，奖励与正确性相关系数约为0.997。

**Decision:** 离线不变量通过，随后进一步简化为Reward v8。

## Reward v8: Binary execution reward

**Motivation:** 只依据最终SQL执行正确性进行训练，避免复杂代理指标误导策略。

**Design:** 执行正确奖励1，执行错误奖励0，不直接奖励结构、工具外观或过程。

**Result:** 1.5B Final Holdout中SFT为47.48%，v8 Step32为55.40%，提升7.91个百分点，McNemar p=0.00000870。

**Decision:** 接受并作为锁定主基线；选择Step32以避开后期策略坍缩。

## Reward v9: Verified process reward

**Motivation:** 缓解Reward v8中大量全错误GRPO组没有信用分配信号的问题。

**Design:** 加入工具顺序、schema检查、协议、执行验证及过程惩罚。

**Result:** Reward信息组率接近100%，但Selection417中v9 Step20为51.08%，低于v8的52.04%；Step32发生协议坍缩。

**Decision:** 拒绝；密集过程奖励产生了可被策略利用的代理目标。

## Reward v10: Correctness-gated process reward

**Motivation:** 保留过程排序能力，同时禁止错误轨迹通过良好过程获得更高奖励。

**Design:** 错误轨迹固定为-0.1且过程bonus为0；只有执行正确轨迹才能获得过程bonus。

**Result:** 扩展SFT同预算Step88相对未扩展Step88从48.20%提高到54.20%，提升6.00个百分点，p=0.00155。

**Decision:** 选择扩展SFT Reward v10 Step88；门控避免v9式协议坍缩，但长期训练仍出现语义坍缩。
