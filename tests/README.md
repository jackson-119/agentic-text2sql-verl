# 测试

统一入口登记14个模块、119项测试：

| 分组 | 内容 | 数量 |
|---|---|---:|
| core | 奖励、SQL评估、评分入口、SFT掩码 | 43 |
| sft | Trainer接口、构建器、Parquet管线 | 18 |
| grpo | 配置准备器、启动器 | 11 |
| algorithms | SC核心、辅助log-prob、IS、累计过程回报 | 37 |
| tools | SQLite工具后端 | 10 |

执行指定组：python scripts/run_tests.py --group algorithms
执行完整清单：python scripts/run_tests.py --group all

原有98项曾完成统一回归；新增GRPO的11项、累计回报的10项分别通过。
本轮仅更新清单，未重新执行119项完整回归。

GRPO组需要Hydra和OmegaConf，训练子进程测试使用模拟对象。
算法组使用CPU张量；工具组需要相应verl工具依赖。
独立Ray worker中重跑的6项评分测试属于已有测试，不重复计数。
真实Tokenizer三轨迹验收另行登记。

累计回报测试保留历史核心语义，包括格式项局部且不受正确性门控，
不意味着新仓库在线事件映射已接入。

参见[测试说明](../docs/testing.md)、
[GRPO入口](../docs/grpo_launcher.md)及[累计回报](../docs/cumulative_credit.md)。
测试通过不等于干净环境安装或完整模型训练复现。
