# SC在线接线补丁

本目录保存历史SC核心接线，不含后加的Actor/q诊断采集。
补丁仅在临时基线副本上验证应用与目标哈希，没有修改训练环境，
没有重新执行运行时测试或训练。

## 基线与应用顺序

verl补丁以verl仓库根目录为根；vLLM补丁以vllm包目录为根。
顺序为verl核心接线、vLLM采样接线；两者齐备后才可启用SC。
预热dummy兼容修复已合入vLLM补丁。

这些基线是历史本地修改版文件，不保证等于记录的Git HEAD或
官方vLLM发行包。必须逐文件匹配清单中的before_sha256；
新文件要求不存在。不匹配时停止，不能强制套用。
不要向当前已打补丁的训练环境再次应用。

此处不提供自动安装或启动命令；发布前仍需完整来源与许可核对。

## 支持范围

固定4个独立辅助样本，保留重复抽样，独立随机数生成器。
实际processed sampling log-probs用于辅助抽样。
candidate slots被复用为私有SC传输协议，不再是通常的top-k输出。

要求非推测采样、FINAL_ONLY输出、非混合greedy/random采样批次。
Agent使用padded、非continuous-token路径。
Actor要求non-fused、remove-padding、无Ulysses序列并行；
SC-only使用vanilla PPO，不叠加IS或distillation。
历史SC运行关闭Actor torch.compile，不宣称支持开启编译。
VERL_SC_ENABLED必须显式传递到相关worker。

## 验证边界

本次验证是补丁文件应用、Python语法及应用后字节哈希。
历史核心单元测试和在线实验已有独立记录，本次没有重跑。
不宣称新仓库的干净环境端到端训练已经复现。

完整文件哈希见experiments/manifests/sc_patch_bundle.json。
