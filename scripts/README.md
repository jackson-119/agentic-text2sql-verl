# 脚本入口

当前提供 run_tests.py，用于分组执行已整理的 CPU 测试。

在仓库根目录运行：

    python scripts/run_tests.py --group core
    python scripts/run_tests.py --group algorithms
    python scripts/run_tests.py --group tools
    python scripts/run_tests.py --group all

默认 core。缺少依赖或任一模块失败时返回非零状态。
此入口不安装依赖、不启动模型或训练。

数据构建、SFT/GRPO 训练、恢复和评测入口尚未迁入。
不收录历史失败重试脚本集合。
