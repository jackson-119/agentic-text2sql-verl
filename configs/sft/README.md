# SFT 训练入口

已迁入 `agentic_text2sql.sft.trainer`，使用 Transformers Trainer 执行
单进程、单卡、BF16 全参数 Assistant-only SFT。

运行模板为 `configs/sft/train_single_gpu.sh`，通过 bash 执行。
模板默认 SDPA；原脚本默认 Flash Attention 2，这是明确的工程配置差异。
显存需求、模型适配及实际训练仍需在线验证。

必填环境变量：

- TEXT2SQL_MODEL_PATH：本地初始化模型目录。
- TEXT2SQL_TRAIN_PARQUET、TEXT2SQL_VAL_PARQUET：两个不同的数据文件。
- TEXT2SQL_SFT_OUTPUT：不存在或为空的输出目录。
- TEXT2SQL_SFT_LR：明确选择的初始学习率。
- CUDA_VISIBLE_DEVICES：只暴露一张 GPU。
- PYTHON_BIN：可选，Python 可执行文件，默认 python。

数据列必须包含 messages_json、tools_json、trajectory_id。
messages_json 中的 trainable 元数据用于控制 Assistant 监督范围。

入口不自动下载模型，但保留历史脚本 trust_remote_code=True 的设置；
只使用可信的本地模型目录。数据加载库可能使用其自身缓存目录。
未设置 HF_DATASETS_CACHE 时，遵循 datasets 的默认缓存配置。

默认 report_to=none，不开启 W&B；设 TEXT2SQL_REPORT_TO=wandb 可开启。
依赖为 torch、transformers、datasets、accelerate；
W&B 和 Flash Attention 2 仅在选用时需要。依赖锁定尚未完成。

训练使用 cosine 调度、warmup_ratio=0.03、weight_decay=0.01、
gradient checkpointing、adamw_torch_fused；最终模型写入 final_hf。
保留最近两个 Trainer checkpoint，不自动选择验证最优模型。

这是新的可配置运行入口，不是已核实的历史启动命令。
历史初始化 checkpoint 和初始学习率仍未完全确认。
本入口暂不提供续训参数，不宣称支持多进程训练。

迁移接口测试不加载模型；实际模型训练与干净环境复现尚待验证。

## 数据构建

已迁入 `agentic_text2sql.sft.builder`，见 [构建说明](../../docs/sft_data_construction.md)。
运行 `python scripts/run_tests.py --group sft` 执行构建器与训练入口接口测试；不会加载模型。
