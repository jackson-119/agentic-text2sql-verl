# 依赖分组与环境记录

pyproject.toml 提供以下可选依赖组：

- algorithms：PyTorch。
- sft：PyTorch、Transformers、datasets、PyArrow、Accelerate。
- tracking：W&B，只有启用对应记录后端时才需要。
- config：Hydra 与 OmegaConf，供配置预检使用。

在自行准备的虚拟环境中，可选择安装：

    python -m pip install -e ".[sft]"

此命令是安装方式说明，不是已通过的干净环境安装方案。
这些依赖未限定完整版本范围；不能据此声称任意最新版本兼容。
应先按硬件选择适合的PyTorch构建，再核对环境。

tools测试仍依赖已核实版本的verl工具接口，本次不自动声明或安装最新版verl。
vLLM与定制框架补丁将在GRPO/SC适配阶段单独约束。
默认SFT模板使用SDPA；Flash Attention 2不是本次可选依赖组的必需项。

[已有环境版本记录](../results/sft/observed_environment.json)
仅记录采集时的软件版本，未覆盖全部传递依赖、系统库、驱动或本地源码补丁。
它不是requirements锁文件，不应直接用于声称完整复现。

## Tokenizer限制

已验证的Qwen2TokenizerFast加载时出现regex警告。
本次未自动应用fix_mistral_regex，掩码对齐通过不等于分词规则已核验。
后续若改变分词规则，须另行比较token ID、掩码和历史实验可比性。
