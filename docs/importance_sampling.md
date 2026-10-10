# Token-level 截断重要性权重

## 本仓库的范围

该模块是对 verl 现有 token-level TIS 权重分支的窄范围参考实现，
不是新提出的算法，也不是完整 rollout correction 框架替代品。

历史实验使用：
token 权重、上限 2、关闭批次归一化、关闭拒绝采样；
GRPO advantage 不做标准差归一化，损失聚合为 token-mean。

## 实际计算顺序

1. 将调用方传入的 log_ratio 截断到源码的数值安全边界。
2. 取指数。
3. 乘二值 response_mask。
4. 权重上截断到 2。
5. detach，阻断权重分支梯度。

不会把权重重新归一化到均值 1。
上截断不是剔除样本：原权重大于 2 时变成 2，而非变成零。
该权重与 PPO 当前/旧策略 ratio 及 PPO clipping 是不同层次的操作。

## 接口限制

token_truncated_is_weights(log_ratio, response_mask, upper_bound=2.0)

输入为形状一致的二维张量。相比历史函数，本接口增加显式输入校验，
拒绝非有限 log-ratio、非二值 mask 和非法上限。
有限有效输入上的权重与已审计的历史 token 分支一致。

没有迁入 sequence 权重、IcePop、批次归一化、拒绝采样或框架指标计算。
本模块不构造 log_ratio，不能据此证明训练端与采样端概率语义一致。

历史采样温度为 0.1，top-p 为 0.95；仅开启 calculate_log_probs
不能证明返回值就是经过全部采样变换后的实际 q。
top-p 的支持集截断、权重截断及前缀分布差异，也不能被称为
完整无偏的目标策略修正。

## 验证

9 项 CPU 测试，加 64 组 float32/float64 原函数分支数值对照。
对照直接执行已审计源码的权重函数，仅将指标计算替换为空返回；
未导入完整 verl，未验证其指标或线上调用链。

保留源文件版权/许可注释，来源哈希登记在 manifest。
这是数值参考实现验证，不是新仓库端到端训练复现。

参见 [策略修正结果](policy_correction_results.md)。

## 运行测试

    PYTHONPATH=src python -m unittest discover -s tests -p 'test_importance_sampling.py' -v
