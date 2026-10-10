# 实验结果

按研究问题组织，只保存小型汇总及来源记录，不保存模型、数据库或完整轨迹。

- [策略修正比较](../docs/policy_correction_results.md)：Control / IS / SC，
  固定 Step120 主比较与探索性最佳 checkpoint。
- [奖励迁移回放](reward_migration/replay_summary.json)：160 条真实训练轨迹。
- [配置与工具预检](reward_migration/configuration_preflight.json)：
  Hydra 组合、包导入和工具异步生命周期；非在线训练复现。

信用分配和 SC 梯度机制结果将在核验后单独登记。
Pilot200 与反复使用的 Selection417 不作为独立最终测试。
