# GRPO配置

baseline_smoke5.yaml来自历史Control，输入和输出通过环境变量指定。
已提供配置准备器和默认Dry-run的显式启动器。

已完成配置Compose、11项入口单元测试、真实配置Dry-run，
以及独立CPU Ray worker的包导入、环境传递和评分测试。
没有验证新入口的GPU训练链路或在线Reward Manager调用。

适配绑定已检查的本地修改版verl，不支持任意上游版本。
累计回报、IS和SC的完整可运行训练配置尚未发布。

参见[配置说明](../../docs/baseline_configuration.md)和
[准备与启动说明](../../docs/grpo_launcher.md)。
