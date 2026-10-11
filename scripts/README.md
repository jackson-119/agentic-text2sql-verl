# 脚本入口

当前提供 run_tests.py，用于分组执行已整理的 CPU 测试。

在仓库根目录运行：

    python scripts/run_tests.py --group core
    python scripts/run_tests.py --group sft
    python scripts/run_tests.py --group grpo
    python scripts/run_tests.py --group algorithms
    python scripts/run_tests.py --group tools
    python scripts/run_tests.py --group all

默认 core。缺少依赖或任一模块失败时返回非零状态。
此入口不安装依赖、不启动模型或训练。

数据构建和训练入口位于 Python 包中，不在本目录重复维护历史脚本：

- [SFT 数据构建](../docs/sft_data_construction.md)：agentic_text2sql.sft.builder。
- [单卡 SFT 模板](../configs/sft/README.md)：agentic_text2sql.sft.trainer。
- [GRPO 准备与启动](../docs/grpo_launcher.md)：agentic_text2sql.training.prepare_grpo 和 launch_grpo。

GRPO 启动器默认 Dry-run；模型训练尚未在新仓库验收。
完整恢复调度和模型批量评测流程尚未统一封装。
不收录历史失败重试脚本集合。
