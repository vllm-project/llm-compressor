# `fp8` Weight and Activation Quantization for Granite 4

`llmcompressor` supports quantizing weights and activations to `fp8` for memory savings and inference acceleration with `vllm`

For Granite 4, in addition to typical `nn.Linear` layers in `mamba` or `mlp` modules, there are three "Linear-like" layers in `GraniteMoeHybridMoe` (`moe` module) that could be quantized as well. Among the three layers, usually `router` should be kept in high precision for accuracy reason. Therefore, users could choose to quantize the other two layers, `input_linear` and `output_linear`, for better model compression.

Note that input_linear and output_linear are `GraniteMoeHybridParallelExperts`, which subclasses `nn.Module` instead of `nn.Linear`, for it needs to store weights in 3D, i.e. [num_experts, out_feat, in_feat]. Because llm-compressor can only handle `nn.Linear` at the moment, our simple workaround would be:
1. **Swap `GraniteMoeHybridParallelExperts` with `GraniteMoeHybridParallelExpertsLinear`**

   The custom class is equivalent to the original one, except it subclasses nn.Linear and stores 2D weights. Moe expert weight tensors will be converted from 3D to 2D, i.e. from [num_experts, out_feat, in_feat] to [num_experts * out_feat, in_feat].
2. **Perform dynamic fp8 quantization**

   The new class is compatible with typical per-channel weight quantization, llm-compressor will be able to identify those layers and process them normally. The resulting scales will have shape of [num_experts * out_feat, 1]
3. **Reshape weights and scales back to 3D before saving the checkpoint**

> `fp8` compuation is supported on Nvidia GPUs with compute capability > 8.9 (Ada Lovelace, Hopper).

## Installation

To get started, install:

```bash
pip install llmcompressor
```

This checkpoint format will need the latest vllm (ver >= 0.10.1.1) to run correctly. Additional dependencies and environment variables needed are:
1. Dependencies:  `vllm>=0.10.1.1, lm_eval>=0.4.9.1, flash-attn=2.7.3, torch>=2.7.1`
2. ENV VAR:  `VLLM_USE_V1=0, VLLM_WORKER_MULTIPROC_METHOD=spawn`

## Quickstart

`granite4_example.py` demonstrates the quantization of `mamba`, `mlp`, and those
"Linear-like" input/output layers with minimal changes to `llm-compressor`.


```bash
python3 granite4_example.py
```

The resulting model `ibm-granite-4-tiny-fp8-dynamic-skipMoeRouter` is ready to be loaded into vLLM.

## Code Walkthough

Now, we will step though the code in the example. There are three steps:
1) Load model
2) Apply quantization
3) Evaluate accuracy in vLLM

### 1) Load Model

Load the model using `AutoModelForCausalLM`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "ibm-granite/granite-4.0-tiny-preview"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2) Apply Quantization

We recommend targeting all `Linear` layers using the `FP8_DYNAMIC` scheme, which uses:
- Static, per-channel quantization on the weights
- Dynamic, per-token quantization on the activations

Since simple PTQ does not require data for weight quantization and the activations are quantized dynamically, we do not need any calibration data for this quantization flow.

Note that we replace the 3D moe expert layers with their 2D equivalent counterpart before quantization and convert them back to 3D before model saving.

```python
from compressed_tensors.utils import replace_module
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

skip_router_only = True  # assume we want to quantize input/output moe layers

ignore_lay = ["lm_head",]
if skip_router_only:
    # swap moe linears to a custom class
    for n, m in model.named_modules():
        if isinstance(m, GraniteMoeHybridParallelExperts):
            new_mod = GraniteMoeHybridParallelExpertsLinear.from_3d_expert(m)
            replace_module(model, n, new_mod)
    ignore_lay += ["re:.*block_sparse_moe.router"]
    SAVE_DIR = "ibm-granite-4-tiny-fp8-dynamic-skipMoeRouter"

# Configure the simple PTQ quantization
recipe = QuantizationModifier(
    targets=["Linear", "GraniteMoeHybridParallelExpertsLinear"],
    scheme="FP8_DYNAMIC",
    ignore=ignore_lay,
)

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Revert weights of MoE experts to 3D format (num_experts, output_size, input_size)
for n, m in model.named_modules():
    if isinstance(m, GraniteMoeHybridParallelExpertsLinear):
        m.to_3d_expert()

# Save the model.
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

We have successfully created an `fp8` model!

### 3) Evaluate Accuracy

Install `vllm` and `lm-evaluation-harness`:

```bash
pip install vllm lm_eval
```

Load and run the model in `vllm` and evaluate accuracy with `lm_eval` on `gsm8k`:

1. **Base model**
```bash
export MODEL=ibm-granite/granite-4.0-tiny-preview
export OPT_FLAGS=tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95,enable_prefix_caching=False,max_model_len=8192
lm_eval --model vllm \
    --model_args pretrained=$MODEL,$OPT_FLAGS,add_bos_token=True \
    --batch_size auto --trust_remote_code --cache_requests true --tasks gsm8k
```
> Note: quantized models can be sensitive to the presence of the `bos` token. `lm_eval` does not add a `bos` token by default, so make sure to include the `add_bos_token=True` argument when running your evaluations.


|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.602|±  |0.0135|
|     |       |strict-match    |     5|exact_match|↑  |0.583|±  |0.0136|

2. **FP8 model**
```bash
export MODEL=$PWD/ibm-granite-4-tiny-fp8-dynamic-skipMoeRouter 
lm_eval --model vllm \
    --model_args pretrained=$MODEL,$OPT_FLAGS,add_bos_token=True \
    --batch_size auto --trust_remote_code --cache_requests true --tasks gsm8k
```

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6164|±  |0.0134|
|     |       |strict-match    |     5|exact_match|↑  |0.5974|±  |0.0135|

We can see the resulting FP8 model look comparable with (and sometimes slightly better than) the baseline.

> NOTE: If running with hf instead of vllm, such as the command below, there will be an error
related to the `weight_scale` when the FP8 ckpt is being used.
`lm_eval --model hf --model_args pretrained=$MODEL --batch_size 16 --trust_remote_code --tasks gsm8k`


### Questions or Feature Request?

Please open up an issue on `vllm-project/llm-compressor`
