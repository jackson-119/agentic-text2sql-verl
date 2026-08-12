# Reward v1：执行结果加权终局奖励

## 状态

- 实验阶段：在线Agentic GRPO
- 入口函数：`compute_score`
- 统一源码：`rewards/source/reward_v1.py`
- 版本副本：`rewards/v1/reward.py`
- SHA256：`ee192db7040f6467452114b798106c85f969d3eb0948f2bea1014e7d9071783f`

## 设计

Reward v1是本项目第一个用于多轮工具Agentic RL实验的奖励函数。它在同一个只读SQLite数据库上分别执行预测SQL和标准SQL，并比较执行结果。

```text
score = 0.90 * execution_correct
      + 0.05 * sql_executable
      + 0.05 * format_compliance
```

奖励由三部分组成：

- `execution_correct`：预测SQL与标准SQL返回等价结果。
- `sql_executable`：提取出的最终SQL能够成功执行。
- `format_compliance`：回答符合严格的`FINAL_SQL:`输出格式。

Agent Loop仍然可以调用`list_tables`、`get_table_schema`和`execute_sql`，但v1不直接奖励中间工具动作、错误恢复或调用效率。

## 实验分析

Reward v1实际完成过五步在线GRPO训练，并非未使用的历史代码。

| 指标 | Step 0 | Step 5 | 变化 |
| --- | ---: | ---: | ---: |
| 执行准确率 | 25.00% | 34.50% | +9.50个百分点 |
| SQL可执行率 | 38.50% | 60.00% | +21.50个百分点 |
| 最终SQL输出率 | 39.00% | 61.50% | +22.50个百分点 |
| 重复execute率 | 72.50% | 35.50% | -37.00个百分点 |
| 工具错误率 | 48.50% | 37.50% | -11.00个百分点 |

结果表明，基于执行结果的终局监督能够快速改善Agent表现。但该奖励较为稀疏，无法直接解决中间工具动作的信用分配问题；Step5仍存在较高的重复执行率和工具错误率。这些问题推动了后续Reward v2和Reward v3对过程奖励的探索。

## 来源确认

当前源码与Reward v2、Reward v3构建前保存的备份具有相同SHA256。训练配置和运行时Reward路由也都指向该文件，因此可以确认它是本项目第一轮在线Reward实验实际使用的Reward v1。
