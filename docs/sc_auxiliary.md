# SC 辅助动作 log-prob 与序列对齐

## 模块边界

algorithms/sc_auxiliary.py 只计算当前策略下辅助动作的平均 log-prob，
不负责构造行为分布 q、采样或传输辅助动作。

从历史实现迁入时，仅移除了 VERL_SC_DIAG_DIR 对应的临时采集钩子。
保留四个辅助样本限制、重复样本、分块计算与梯度检查点逻辑。

## 输入约定

- logits：[N, vocabulary_size]，已按 Actor 的温度约定处理。
- aux_ids：[N, 4]，int64；位于 token j 的辅助动作使用位置 j-1 的 logits。
- offsets：packed 序列边界，覆盖 [0, N]，必须严格递增。
- 缺失位置必须使用四个 -1，不能部分缺失。
- 当前接口要求调用方提供合法整数序列边界；不宣称完整校验所有错误输入。

每条序列的最后一个 logits 位置不会使用下一条序列的辅助动作。
因此，序列边界置空不能省略。

## 数值与梯度

有放回采样产生的重复动作按出现次数计入平均值，不能去重。
float64 输入保持 float64；其他训练输入使用 float32 计算归一化。
有效位置分块计算，并在需要梯度时使用非重入梯度检查点。
缺失位置输出零，不产生该局部项的梯度。

## 验证范围

8 项 CPU 测试覆盖独立逐序列参考计算、梯度、因果偏移、
序列边界、重复动作、缺失位置和错误输入。

本轮没有验证实际 vLLM 采样、Agent 数据传输或分布式 Actor 接线。
与 SC 修正核心分别通过测试，不代表新仓库已完成端到端训练验证。

参见 [SC 修正核心](score_centering.md)。

## 运行测试

    PYTHONPATH=src python -m unittest discover -s tests -p 'test_sc_auxiliary.py' -v
