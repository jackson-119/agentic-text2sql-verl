# 语义扩展 SFT 与监督掩码

## 数据与历史产物

历史验收记录包含 5000 条训练轨迹、500 条验证轨迹，
分别覆盖 119 与 21 个数据库，SFT 内部数据库集合和 source ID 不重叠。
训练轨迹中 4488 条为标准成功轨迹，512 条为 schema 错误恢复轨迹。

初版验收失败，第二版按 source ID 等口径验收通过；
没有证据表明这是删除重复数据后得到的结果。
第二版记录 4 个语义重复项，并将其视为提示信息。

报告的 source_coverage 为 0.7857142857142857，
分母与计算实现尚未迁入，不将它改写为“训练集覆盖率”。
SFT 内部划分不重叠，不代表已验证与 Pilot200 等所有后续评测集无交叉。

## 监督构造

只监督 role=assistant 且 trainable 不为假的消息。
先从消息中移除 trainable 字段，再调用 chat template。

利用 add_generation_prompt=True 得到 Assistant 生成头的终点，
从该位置到当前 Assistant 消息结束的 token 参与监督。
因此生成头不监督；模板附带结束标记是否纳入，取决于模板实际 token 区间。
User、System、Tool 和不训练的 Assistant 消息均保留为上下文，label=-100。

逐轮前缀和完整对话前缀必须一致，否则报错。
超长序列报错，不静默截断；没有监督 token 的样本也报错。
掩码函数、Dataset、padding collator 与单卡 Trainer 入口已迁入；在线模型训练尚未在新入口中验证。

## 已确认训练记录

- 1 epoch，313 个训练 step。
- trainer_state.train_batch_size=1；不将其等同于全局有效 batch size。
- logging/eval/save 间隔为 10/50/100。
- 历史训练摘要 train_loss=0.09793120803543554。
- 摘要运行时间约 1911 秒，不作为不同硬件间的性能基准。

源码使用 Transformers Trainer，没有配置 LoRA 或冻结层。
实际初始学习率和完整启动命令尚未确认，不能照抄 argparse 默认参数。
数据 manifest 的 model_path 不足以单独证明实际训练初始化路径。

模型 config 的 vocab_size=151936；
tokenizer JSON 的不同 token ID 数及最大 ID 加一均为 151665。
这些字段含义不同，不混用为同一个词表统计。

## 验证与复现状态

历史 loss mask 报告抽查 32 条样本并通过。
新仓库新增 9 项合成 tokenizer 契约测试，不代表真实模板端到端回放。
数据构建器与单卡 SFT 入口均已迁入，并提供显式训练参数模板；历史数据逐条重建、模型训练和干净环境复现尚未完成。

本 SFT 模型目录被后续已整理的 GRPO 配置引用为初始化模型。
这一配置关联不等于本轮重新验证了模型权重或重新运行训练。

- [历史精简证据](../results/sft/semantic_expanded_summary.json)
- [来源记录](../experiments/manifests/sft_masking_source.json)

## 新训练入口

参见 [运行配置说明](../configs/sft/README.md)。该入口要求显式学习率，拒绝非空输出目录，默认关闭 W&B。增加的随机种子初始化及模板默认 SDPA 均属于迁移调整，不宣称与历史运行逐位等价。

## 新增数据管线验收

- 合成 SQLite → 构建器 → 真实 Parquet → Dataset → Collator 已通过。
- 本地 Qwen2TokenizerFast 在标准、恢复和验证三条合成轨迹上通过字符 offset 与 labels 的交叉核对。
- 错误恢复额外增加 68 个上下文 token，监督 token 仍为 79。
- Tokenizer regex 警告尚未解决；本次未改变历史加载参数。

这些测试不是历史数据重建，也未加载模型权重或执行训练。
详见 [迁移验收](../results/sft/migration_validation.json)。
