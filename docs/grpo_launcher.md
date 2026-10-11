# GRPO配置准备与启动

## 范围

原始V10标量奖励，单机单GPU全新运行；不启用IS、SC或累计回报。
当前不支持断点续训。适配通过哈希绑定已检查的本地修改版verl入口。

## 准备器

模块：agentic_text2sql.training.prepare_grpo

必需参数：--repo、--verl-root、--run-dir、--ray-temp-dir。

必需环境变量：TEXT2SQL_MODEL_PATH、TEXT2SQL_CRITIC_MODEL_PATH、
TEXT2SQL_TRAIN_PARQUET、TEXT2SQL_DEV_PARQUET、SPIDER_DATABASE_ROOT。

运行目录应位于仓库和输入目录之外，且尚不存在；Ray目录使用独立新路径。
输出tools.yaml、train.yaml和preparation.json。
准备器显式配置driver/worker环境、关闭SC与诊断开关，
将Ray设置放入实际使用的ray_kwargs.ray_init。
存在冲突时停止，未使用的必填占位符保留。
准备阶段不启动Ray或模型。

## 启动器

模块：agentic_text2sql.training.launch_grpo

默认检查命令：

    python -m agentic_text2sql.training.launch_grpo --preparation "$PREPARATION_JSON"

只有增加--execute才启动训练，并要求显式设置单GPU的CUDA_VISIBLE_DEVICES。
调用前需安装当前包或将仓库src加入PYTHONPATH。

启动器校验配置、模板、训练入口和环境声明。
输出launch_claim.json、training.log和launch_result.json，
分别记录独占启动计划、子进程日志和退出状态。
同一目录拒绝重复启动；失败后先查看日志，不自动清理或恢复。
强制终止可能留下启动记录但无退出报告，不能视为成功。
远程训练应使用持久终端会话；启动器不是后台任务管理器。

## 验证范围

准备器5项、启动器6项单元测试和真实配置Dry-run已通过。
独立CPU Ray worker正确导入新包、收到环境变量，
并通过已有6项评分入口测试；没有直接验证在线Reward Manager。

历史算法结果来自原训练工程，不因仓库整理而重跑。
新入口GPU端到端运行和干净环境复现仍未验证。

准备记录是可信本地输入，不是安全隔离边界。
工具配置哈希在启动计划生成时采集；未锁定全部依赖、模型和数据。
本机路径留在外部运行报告中，不纳入发布结果。

参见[验证摘要](../results/grpo/migration_validation.json)。
