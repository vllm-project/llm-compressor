<div align="center">

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="docs/assets/llmcompressor-icon-name-dark.png"
  >
  <source
    media="(prefers-color-scheme: light)"
    srcset="docs/assets/llmcompressor-icon-name-light.png"
  >
  <img
    src="docs/assets/llmcompressor-icon-name-light.png"
    alt="LLM Compressor"
    width="420"
  >
</picture>

[![docs](https://img.shields.io/badge/docs-LLM--Compressor-blue)](https://docs.vllm.ai/projects/llm-compressor/en/latest/) [![PyPI](https://img.shields.io/pypi/v/llmcompressor.svg)](https://pypi.org/project/llmcompressor/)

</div>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

`llmcompressor` 是一个快速、高效且易于使用的算法库，专注于为 vLLM 推理部署进行大模型优化与压缩，具备以下核心能力：

* **完备的量化算法与变换支持**：涵盖权重（Weight）、激活值（Activation）、KV Cache 以及注意力机制（Attention）的各类量化算法
* **与 Hugging Face 无缝集成**：原生兼容 Hugging Face 格式模型与开源社区仓库
* **vLLM 原生兼容**：模型以 `compressed-tensors` 格式存储，与 vLLM 高性能推理引擎开箱即用无缝适配
* **高效硬件支持**：支持 DDP（分布式数据并行）与磁盘卸载（Disk Offloading），以极高的硬件效率压缩超大规模模型

**✨ 阅读官方发布博客 [点击此处](https://neuralmagic.com/blog/llm-compressor-is-here-faster-inference-with-vllm/)！✨**

<p align="center">
   <img alt="LLM Compressor Flow" src="https://github.com/user-attachments/assets/adf07594-6487-48ae-af62-d9555046d51b" width="80%" />
</p>

---

📊 欢迎填写我们的 [1 分钟用户调研问卷](https://red.ht/llm-compressor-user-survey)，帮助我们持续改进

💬 欢迎加入 [vLLM 官方社区 Slack](https://inviter.co/vllm-slack)，在以下专属频道分享您的问题、想法与建议：

- `#sig-quantization`
- `#llm-compressor`

---
## 🚀 最新动态 (What's New!)

LLM Compressor 迎来了多项重磅更新！如需更深入了解，请参阅 [LLM Compressor 概览演示文档](https://docs.google.com/presentation/d/1WNkYBKv_CsrYs69lb7bJKjh2dWt8U1HXUw7Gr4Wn3gE/edit?usp=sharing)。

部分重磅全新功能包括：

* **MXFP4 量化 GLM-5.3**：红帽 AI 团队（Red Hat AI）构建了 GLM-5.3 的 MXFP4 量化权重。Transformer 模块中的线性算子被量化至 MXFP4，同时 MoE 路由层、嵌入层、DSA 索引器和输出头保留原始精度，以保证模型精度的恢复。
  - [RedHatAI/GLM-5.3-MXFP4](https://huggingface.co/RedHatAI/GLM-5.3-MXFP4)
  - [GLM-5.3 MXFP4 示例](examples/model_free_ptq/glm_5_3_mxfp4.py)
* **NVFP4 量化 GLM 5.3-Flash**：GLM-5.3-Flash 的 NVFP4 量化权重。专家层（Expert layers）已量化至 NVFP4，MTP 层按 Block 逐块量化至 FP8。
  - [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)
* **Qwen3.8 NVFP4、FP8 与 INT4 量化权重**：红帽 AI 团队构建了 Qwen3.8-2.4T-A95B 的 NVFP4 与 FP8 量化权重，以及 Qwen3.8-27B 的 INT4 权重。特别值得关注的是，`Qwen3.8-2.4T-A95B-NVFP4-REAP-25` 将 REAP 专家剪枝与 NVFP4 量化相结合——在量化之前剪去 25% 最不重要的专家层，在保持模型精度恢复的同时进一步大幅降低显存需求。
  - 模型权重：
    - [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)
    - [RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-REAP-25](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-REAP-25)
    - [RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-FP8](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-FP8)
    - [RedHatAI/Qwen3.8-2.4T-A95B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-NVFP4)
    - [RedHatAI/Qwen3.8-2.4T-A95B-FP8](https://huggingface.co/RedHatAI/Qwen3.8-2.4T-A95B-FP8)
  - 示例脚本：
    - [Qwen3.8-2.4T-A95B NVFP4+FP8 示例](examples/quantizing_moe/qwen_3_8_example.py)
    - [Qwen3.8-2.4T-A95B REAP + NVFP4 示例](examples/reap_expert_pruning/qwen38_example.py)
    - [Qwen3.8-27B INT4 示例](examples/quantization_w4a16/qwen3_8_gptq_awq_example.py)
* **Nemotron 3.5 Lightning FP8 量化权重**：红帽 AI 团队使用基于 GPTQ 的 FP8 量化构建了 [Nemotron 3.5 Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) 的 FP8 量化权重。
  - [RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8](https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8)
  - [Nemotron 3.5 Lightning FP8 示例](examples/quantization_w8a8_fp8/nemotron_3_5_lightning_example.py)
* **Muse-Glimmer-30B FP8、NVFP4 与 INT4 量化权重**：红帽 AI 团队构建了 [Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) 的 FP8、NVFP4 与 INT4 权重，支持在单张 GPU 上部署该多模态大模型。
  - [RedHatAI/Muse-Glimmer-30B-FP8-block](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-FP8-block)
  - [RedHatAI/Muse-Glimmer-30B-NVFP4](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-NVFP4)
  - [RedHatAI/Muse-Glimmer-30B-W4A16](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-W4A16)
  - [Muse-Glimmer FP8_Block 示例](examples/model_free_ptq/muse_glimmer_fp8_block.py)
* **Kimi-K3 NVFP4 与 FP8 量化权重**：红帽 AI 团队构建了 Kimi-K3 的 NVFP4 与 FP8 量化权重。
  - [RedHatAI/Kimi-K3-NVFP4](https://huggingface.co/RedHatAI/Kimi-K3-NVFP4)
  - [RedHatAI/Kimi-K3-FP8-BLOCK](https://huggingface.co/RedHatAI/Kimi-K3-FP8-BLOCK)
* **Hy3 NVFP4+FP8 量化权重**：红帽 AI 团队构建了 [Hy3](https://huggingface.co/tencent/Hy3) 的量化权重，将 MoE 层的 NVFP4 量化与注意力层的 FP8 量化相结合，在维持精度恢复的同时大幅减少了显存占用。
  - [RedHatAI/Hy3-NVFP4-FP8](https://huggingface.co/RedHatAI/Hy3-NVFP4-FP8)
  - [Hy3 量化示例](examples/quantization_w4a4_fp4/hy3_example.py)
* **GLM-5.2 NVFP4+FP8 示例与权重**：红帽 AI 团队利用 DDP + 磁盘卸载在不到 2 小时内构建了 [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) 的量化权重。全精度模型需要 1.6TB 显存，但通过对 MoE 层进行 NVFP4 量化以及注意力层进行 FP8 量化，模型体积缩减了 >70%，同时在 GPQA 评测基准上保持了前沿的精度恢复水平。
  - [RedHatAI/GLM-5.2-NVFP4-FP8](https://huggingface.co/RedHatAI/GLM-5.2-NVFP4-FP8)
  - [GLM-5.2 示例脚本](examples/quantizing_moe/glm5_example.py)
* **REAP 专家剪枝修改器 (REAP Expert Pruning Modifier)**：[REAP](https://arxiv.org/pdf/2510.13999) 通过结构化去除各层中相关性较低的专家模块，降低运行混合专家（MoE）模型的显存需求。REAP 通过校准前向传播数据计算出的显著性指标来衡量相关性，能够在达到用户指定的专家稀疏度目标的同时，最大程度减小剪枝对精度的影响。修改器实现位于 [`modifiers/pruning/reap`](src/llmcompressor/modifiers/pruning/reap)，可作为实现其他专家剪枝算法的参考模板。示例与更多文档见下方链接：
  - [REAP 剪枝说明文档](examples/reap_expert_pruning/README.md)
  - [REAP 剪枝 Qwen/Qwen3-30B-A3B-Instruct-2507 至 25% 稀疏度](examples/reap_expert_pruning/reap_qwen3_30b.py)
  - [REAP 剪枝 moonshotai/Moonlight-16B-A3B-Instruct 至 25% 稀疏度](examples/reap_expert_pruning/reap_moonlight_16b.py)



### 支持的精度与类型 (Supported Precisions and Types)
* 激活值量化：W8A8（int8 与 fp8）、W4AFP8、Microscale（NVFP4、MXFP4、MXFP8）
* 混合精度：W4A16、W8A16、MXFP8A16、MXFP4A16、NVFP4A16
* 注意力与 KV Cache 量化：FP8、NVFP4
* 低比特 / 任意比特量化：WNA4、WNA8、WNA16

### 支持的算法 (Supported Algorithms)
* 基础训练后量化 (Simple PTQ)
* GPTQ
* AWQ
* SmoothQuant
* AutoRound
* 基于旋转的算法 (Rotation-based: SpinQuant, QuIP)
* REAP 专家剪枝 (REAP expert pruning)

### 循序渐进量化模型 (Quantizing your model, step-by-step)

关于如何选择量化方案、算法及其具体应用场景的详细信息，请参阅我们的[分步模型压缩指南 (Step-by-step compression guide)](https://docs.vllm.ai/projects/llm-compressor/en/latest/steps/choosing-model/)。

有关 LLM Compressor 功能特性的更多信息，亦可查阅我们的[用户指南 (User Guides)](https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/entrypoints/) 与[常见问题解答 (FAQ)](https://docs.vllm.ai/projects/llm-compressor/en/latest/faq/faq/)。


## 安装指南 (Installation)

```bash
pip install llmcompressor
```

## 快速上手 (Get Started)

### 端到端示例 (End-to-End Examples)

使用 `llmcompressor` 应用量化算法：

### 权重与激活值量化 (Weight and Activation Quantization)
* [激活值量化至 `int8`](examples/quantization_w8a8_int8/README.md)
* [激活值量化至 `fp8`](examples/quantization_w8a8_fp8/README.md)
* [激活值量化至 MXFP8](examples/quantization_w8a8_mxfp8)
* [激活值量化至 `fp4` (NVFP4)](examples/quantization_w4a4_fp4)
* [激活值量化至 `fp4` (MXFP4)](examples/quantization_w4a4_mxfp4)
* [使用 AutoRound 进行 `fp4` 激活值量化](examples/autoround/quantization_w4a4_fp4/README.md)
* [激活值量化至 `fp8` 且权重截断至 `int4`](examples/quantization_w4a8_fp8)

### 纯权重量化 (Weight Only Quantization)
* [纯权重导出为 `fp4`（NVFP4 格式）](examples/quantization_w4a16_fp4/nvfp4)
* [纯权重导出为 `fp4`（MXFP4 格式）](examples/quantization_w4a16_fp4/mxfp4)
* [使用 GPTQ 进行 `int4` 纯权重量化](examples/quantization_w4a16/README.md)
* [使用 AWQ 进行 `int4` 纯权重量化](examples/awq/README.md)
* [使用 AutoRound 进行纯权重量化 (`wNa16`)](examples/autoround/quantization_wNa16/README.md)

### 注意力机制与 KV Cache 量化 (Attention and KV Cache Quantization)
* [KV Cache 量化至 `fp8`](examples/quantization_kv_cache/README.md)
* [基于 Per-head 的 KV Cache `fp8` 量化](examples/quantization_kv_cache/llama3_fp8_head_kv_example.py)
* [注意力机制量化至 `fp8`](examples/quantization_attention/README.md)
* [结合 SpinQuant 的注意力机制 `NVFP4` 量化（实验性功能）](experimental/attention/README.md)

### 特定架构量化 (Architecture-Specific Quantization)
* [量化 MoE 混合专家大语言模型](examples/quantizing_moe/README.md)
* [量化视觉-语言多模态大模型 (VLM)](examples/multimodal_vision/README.md)
* [量化音频-语言多模态大模型 (ALM)](examples/multimodal_audio/README.md)

### 非均匀量化 (Non-Uniform Quantization)
* [模型非均匀混合精度量化 (Quantizing Models Non-uniformly)](examples/quantization_non_uniform/README.md)

### 超大模型量化支持 (Big Model Quantization Support)
* [利用时序依次加载（Sequential Onloading）量化超大模型](examples/big_models_with_sequential_onloading/README.md)
* [利用磁盘卸载（Disk Offloading）量化超大模型](examples/disk_offloading/README.md)

### 无需 HF 模型定义的量化 (Model-Free Definition Quantization)
* [在没有 Hugging Face 模型定义的情况下直接量化模型](examples/model_free_ptq/README.md)

### DDP 分布式量化 (DDP Quantization)
* [结合 GPTQ 的分布式数据并行量化 (Distributed data parallel quantization)](examples/quantization_w4a16/llama3_ddp_example.py)


## 快速导览 (Quick Tour)
下面演示如何使用 `Round-to-Nearest` (RTN) 算法对 `Qwen3-30B-A3B` 的权重与激活值进行 FP8 量化。

请注意，示例中的模型可以替换为任意本地或远端兼容 Hugging Face 的权重，并且可以调整 `recipe` 参数以适配不同的量化算法或格式。

### 应用量化 (Apply Quantization)
通过选择量化算法并调用 `oneshot` API 来应用量化。

```python
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3-30B-A3B"

# 加载模型
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 配置量化算法与方案
# 本例中执行以下操作：
#   * 使用 block_size=128 的 RTN 算法将权重参数量化为 FP8
#   * 在推理时将激活值动态量化为 FP8
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=["lm_head", "re:.*mlp.gate$"],
)

# 应用量化
oneshot(model=model, recipe=recipe)

# 验证量化模型的文本生成输出是否正常
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# 以 compressed-tensors 格式保存至磁盘
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-BLOCK"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

### 使用 vLLM 进行高性能推理 (Inference with vLLM)

由 `llmcompressor` 创建的模型权重可以直接加载并运行在 `vllm` 中：

安装：

```bash
pip install vllm
```

运行：

```python
from vllm import LLM
model = LLM("Qwen/Qwen3-30B-A3B-FP8-BLOCK")
output = model.generate("My name is")
```

## 问题反馈与参与贡献 (Questions / Contribution)

- 如有任何疑问或需求，欢迎提交 [GitHub Issue](https://github.com/vllm-project/llm-compressor/issues)，我们将持续补充相关示例或文档说明。
- 我们非常欢迎对代码、示例、生态集成和文档的贡献，以及 Bug 反馈和新特性建议！[了解如何参与贡献请查阅 CONTRIBUTING.md](CONTRIBUTING.md)。

## 引用说明 (Citation)

如果您在学术研究或工程项目中使用了 LLM Compressor，欢迎引用：

```bibtex
@software{llmcompressor2024,
    title={{LLM Compressor}},
    author={Red Hat AI and vLLM Project},
    year={2024},
    month={8},
    url={https://github.com/vllm-project/llm-compressor},
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月17日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
