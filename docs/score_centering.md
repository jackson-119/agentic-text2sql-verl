# Score Centering 核心

## 接口

apply_sc_correction(pg_loss, log_prob, mean_logp, response_mask)

- pg_loss：已聚合的标量 PPO policy loss。
- log_prob：实际参与该损失计算的当前策略动作 log-prob。
- mean_logp：同一前缀下独立辅助动作的平均当前策略 log-prob。
- response_mask：Assistant token 掩码，不包括工具返回和 padding。

本核心仅依赖 PyTorch，不依赖 verl 或 vLLM。
辅助动作由上游从实际行为分布 q 中有放回采样；
核心不构造 q，也不能自行验证前缀对齐或采样独立性。
mean_logp 必须保留到当前 Actor 参数的梯度。

## 计算方式

令 c = stop_gradient(d pg_loss / d log_prob)，m = mean_logp。

修正后损失：
pg_loss - sum(c * (m - stop_gradient(m)))

附加项前向值为零，反向梯度为负的辅助 score 均值乘以实际 PPO 系数。
系数包含实际裁剪、聚合和上游损失缩放，
因此不能再次乘 advantage 或再次除以 token 数。

这不是奖励中心化，也不是对辅助动作重新计算 PPO clipping。
辅助动作全部等于实际动作时，对应 score 项抵消。
不能通过去重或强制换动作来消除这一现象，否则会改变估计器。

## 接线边界

历史 Actor 接线仅支持 vanilla PPO，并拒绝同时启用 IS。
修正发生在加入熵和 KL 损失之前，仅作用于 policy-gradient 项。
这些模式限制由 Actor 适配层实施，不由此独立函数检查。

本轮只原样迁入修正核心。
辅助采样、packed 序列对齐、Actor log-prob 计算及框架接线，
尚未在新仓库中完整迁入。

## 验证与限制

CPU 测试覆盖独立 PPO 参考目标的系数、前向不变性、解析 logit 梯度、
普通与 dual clipping、掩码、重复辅助动作以及错误输入。

这不是实际 verl PPO 损失的完整重建，也不是分布式 Actor、
参数更新或新仓库端到端训练验证。
停止梯度系数只用于一阶梯度修正，不声称保留原目标的高阶导数。

SC 不构成完整无偏的目标策略修正。
机制成立不等于执行准确率提升，参见
[Control / IS / SC 结果](policy_correction_results.md)。

## 运行测试

在仓库根目录、已安装 PyTorch 的环境中运行：

    PYTHONPATH=src python -m unittest discover -s tests -p 'test_score_centering.py' -v
