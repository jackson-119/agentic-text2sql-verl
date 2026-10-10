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
迁入的是掩码函数，padding collator 与完整 Trainer 尚未迁入。

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
构建数据的代码、完整 SFT 训练入口及干净环境复现尚未完成迁入。

本 SFT 模型目录被后续已整理的 GRPO 配置引用为初始化模型。
这一配置关联不等于本轮重新验证了模型权重或重新运行训练。

- [历史精简证据](../results/sft/semantic_expanded_summary.json)
- [来源记录](../experiments/manifests/sft_masking_source.json)
