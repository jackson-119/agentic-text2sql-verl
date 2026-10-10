# 数据库工具环境

## 职责

`SpiderDatabaseTool` 是项目的 SQLite 工具适配器，基于 verl BaseTool。
多轮调度由 verl Agent Loop 提供，不在本模块重新实现。

提供三个动作：
- list_tables：列出数据库表；
- get_table_schema：按表查询列、主键、外键和少量示例行；
- execute_sql：执行查询，返回列名、结果行及截断标记。

Schema 并非一次性全部检索；每次 get_table_schema 接收一个 table_name。
是否再次查询或输出最终答案由模型及 Agent Loop 决定。

## 状态与反馈

实例保存 db_id 与数据库路径。数据库按只读方式打开，并设置 query_only。
工具执行返回 ToolResponse、数值0.0及工具成功/动作指标。
这里的0.0不是最终任务奖励，任务评分在独立奖励模块中完成。

## 当前迁入范围

源文件原样迁入，未改变行为。依赖已有 verl 环境。
数据库不随仓库分发，来源和授权另行记录。

新增测试使用临时构造的SQLite数据库，仅验证同步查询后端。
尚未验证BaseTool初始化、create/release、异步调用及AgentLoop集成。

## 已知限制

- SQL关键词检查基于正则，可能误拒绝字符串或标识符中的关键词。
- 查询超时使用SQLite进度回调，不是独立进程级硬超时。
- 当前防护适用于受控实验，不宣称是面向任意不可信数据库的完整安全沙箱。
- 本轮没有调整这些历史行为，避免仓库整理引入算法或环境变化。
