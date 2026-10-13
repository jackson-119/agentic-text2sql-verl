# 基线配置说明

配置来自已运行的Control五步实验。
训练算法、预算、采样、优化器和编译设置未主动调整；
模型与数据路径、工具类、奖励入口、输出位置和监控标签已经迁移。

## 必填环境变量

- TEXT2SQL_REPO_ROOT：本仓库绝对路径
- TEXT2SQL_MODEL_PATH：初始SFT模型
- TEXT2SQL_TRAIN_PARQUET：训练数据
- TEXT2SQL_DEV_PARQUET：Pilot开发数据
- TEXT2SQL_RUN_DIR：独立运行输出目录
- TEXT2SQL_RAY_TEMP_DIR：独立且较短的Ray临时目录
- SPIDER_DATABASE_ROOT：Spider database目录
- TEXT2SQL_TOOL_CONFIG：解析后工具配置的绝对路径

若原始tokenizer或Critic路径不同于Actor模型，额外变量记录在
experiments/manifests/baseline_config_source.json中。
保留Critic及skip组件配置，不改变其启用状态。

## 工具模板必须预先解析

configs/evaluation/spider_tools.yaml含oc.env插值。
不得假定verl工具加载器会自动解析它。
启动脚本应先用OmegaConf解析成运行目录下的配置文件，
再把该文件路径赋给TEXT2SQL_TOOL_CONFIG。

## 分布式可见性

新包及数据库环境变量必须对Ray、奖励和采样进程可见。
只在启动终端设置变量，不等于已经证明Worker收到变量。

## 验证状态

本轮只检查模板生成、白名单修改范围和已知绝对路径。
未验证Hydra Compose、Worker导入、Prometheus文件使用或在线训练。
配置扫描也不能替代发布前的完整敏感信息审查。
