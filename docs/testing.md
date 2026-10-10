# 测试与依赖

## 分组入口

在仓库根目录运行：

    python scripts/run_tests.py --group core
    python scripts/run_tests.py --group algorithms
    python scripts/run_tests.py --group tools
    python scripts/run_tests.py --group all

默认运行 core。任一测试模块失败时，入口返回非零退出码。
缺少依赖不会被当作通过或静默跳过。

- core：奖励组合、SQL 执行评估、独立标量评分入口。
- algorithms：SC 梯度修正、辅助 log-prob、token-level IS 权重。
- tools：数据库工具后端测试。

每个测试文件运行于独立子进程，避免模块导入状态互相干扰。
入口显式使用新仓库 src，并隐藏 CUDA 设备。
它不会安装依赖、启动训练或主动启动 Ray。

## 依赖边界

- Python：项目声明为 3.10 及以上；历史验证环境使用 Python 3.10。
- core：当前组件使用 Python 标准库和 SQLite。
- algorithms：需要 PyTorch。已有验证环境为 torch 2.10.0+cu129，
  本组只执行 CPU 张量测试；这不是跨版本兼容性声明。
- tools：需要可导入的 verl 及其工具接口依赖。
- 配置预检还需要 Hydra 与 OmegaConf。

尚未建立独立、干净环境的完整依赖锁定。
不要把服务器已有环境中的测试通过，写成 pip 安装后即可完整训练。
不建议为了运行工具测试，直接安装未经版本核对的最新版 verl。

## 测试边界

算法单元测试不加载语言模型。
数据库测试使用测试夹具，不要求读取 Spider 原始数据库。
数据库后端测试不替代 Agent 多轮在线链路测试。

历史配置预检另行验证过工具异步生命周期；
统一入口当前并未将该预检自动化为在线集成测试。

如果新增测试文件，应同步维护 scripts/run_tests.py 的明确分组。
