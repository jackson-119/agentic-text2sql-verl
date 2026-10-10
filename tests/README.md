# 测试

当前统一入口包含 8 个模块、80 项测试：

| 分组 | 内容 | 测试数 |
|---|---|---:|
| core | 奖励组合、SQL 评估、评分入口、SFT 掩码 | 43 |
| algorithms | SC 核心、辅助 log-prob、token-level IS | 27 |
| tools | SQLite 工具后端 | 10 |

执行方法见 [测试文档](../docs/testing.md)。

这些测试已在原服务器已有环境中通过。
SFT 掩码使用合成 tokenizer；算法测试使用 CPU 张量。
不把这些结果称为真实模型训练、完整分布式接线或干净安装验证。
