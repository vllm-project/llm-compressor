from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridParallelExperts,
)
from compressed_tensors.utils import replace_module
from llmcompressor import oneshot
from llmcompressor.modeling.granite4 import GraniteMoeHybridParallelExpertsLinear
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

"""
There are three "Linear-like" layers in Granite4's GraniteMoeHybridMoe (moe module) that could be
quantized. Among the three layers, usually "router" should be kept in high precision, therefore,
user could choose whether to quantize the other two layers, .input_linear and .output_linear.

This example demonstrates the quantization of those "Linear-like" input/output layers with minimal
changes in llm-compressor.

Note that input_linear and output_linear are `GraniteMoeHybridParallelExperts`, which subclasses
nn.Modules instead of nn.Linear, for it needs to store weights in 3D, i.e. [num_experts, out_feat,
in_feat]. Because llm-compressor can only handle nn.Linear at the moment, our simple workaround
would be:
1. Swap `GraniteMoeHybridParallelExperts` with `GraniteMoeHybridParallelExpertsLinear`
   The custom class is equivalent to the original one, except it subclasses nn.Linear and stores
   2D weights. Moe expert weight tensors will be converted from 3D to 2D, i.e. from [num_experts,
   out_feat, in_feat] to [num_experts * out_feat, in_feat].
2. Perform dynamic fp8 quantization
   The new class is compatible with typical per-channel weight quantization, llm-compressor will
   be able to identify those layers and process them normally. The resulting scales will have shape
   of [num_experts * out_feat, 1]
3. Reshape weights and scales back to 3D before saving the checkpoint

NOTE This checkpoint format will need latest vllm (ver >= 0.10.1.1) to run correctly.
Test settings:
1. DEP VERSION:  vllm=0.10.1.1, lm_eval=0.4.9.1, flash-attn=2.7.3, torch=2.7.1
2. ENV VAR:  VLLM_USE_V1=0, VLLM_WORKER_MULTIPROC_METHOD=spawn
3. device: H100-80G

Results:
1. base model

lm_eval --model vllm  --model_args pretrained=ibm-granite/granite-4.0-tiny-preview,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95,enable_prefix_caching=False,max_model_len=8192 --batch_size auto --trust_remote_code --cache_requests true --tasks gsm8k

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.602|±  |0.0135|
|     |       |strict-match    |     5|exact_match|↑  |0.583|±  |0.0136|

2. FP8 version

lm_eval --model vllm  --model_args pretrained=gr4_fp8_skipRouter_lin_exp3d,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95,enable_prefix_caching=False,max_model_len=8192 --batch_size auto --trust_remote_code --cache_requests true --tasks gsm8k
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6073|±  |0.0135|
|     |       |strict-match    |     5|exact_match|↑  |0.5921|±  |0.0135|


If running with hf instead of vllm, such as the command below, there will be an error message
related to the "weight_scale" when the FP8 ckpt is being used.

lm_eval --model hf  --model_args pretrained=ibm-granite/granite-4.0-tiny-preview,dtype=auto --batch_size 16 --trust_remote_code --tasks gsm8k

"""

MODEL_ID = "ibm-granite/granite-4.0-tiny-preview"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

skip_router_only = True  # assume we want to quantize input/output moe layers

if skip_router_only:
    for n, m in model.named_modules():
        if isinstance(m, GraniteMoeHybridParallelExperts):
            new_mod = GraniteMoeHybridParallelExpertsLinear.from_3d_expert(m)
            replace_module(model, n, new_mod)
            print(f"Replaced {n}")
    ignore_lay = ['re:.*lm_head', 're:.*block_sparse_moe.router']
    SAVE_DIR = "gr4_fp8_skipRouter_lin_exp3d"
else:
    # Skip all .input_linear, .output-linear, and router layers.
    ignore_lay = ['re:.*lm_head', 're:.*block_sparse_moe']
    SAVE_DIR = "gr4_fp8_skipMoe_lin"

recipe = QuantizationModifier(
    targets=["Linear", "GraniteMoeHybridParallelExpertsLinear"],
    scheme="FP8_DYNAMIC",
    ignore=ignore_lay)

# Apply quantization and save in compressed-tensors format.
# NOTE Do NOT save the model using oneshot(..., output_dir=SAVE_DIR) here as it will trigger
#   conversion of weights from BF16 to FP8 and subsequently cause dtype mismatch in the following
#   generation test. For example, F.linear(x, W) in forward() will throw errors as x is in BF16 but
#   W is in FP8.
oneshot(model=model, recipe=recipe)

# Confirm generations of the quantized model look sane.
print("After module swapping")
print(model)
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("What is your favorite TV show?", return_tensors="pt").input_ids.to('cuda')
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# Revert weights of MoE experts to 3D format (num_experts, output_size, input_size)
for n, m in model.named_modules():
    if isinstance(m, GraniteMoeHybridParallelExpertsLinear):
        # NOTE: can assert type != "meta" instead, which is sign of offloading
        assert  m.weight.device.type == "cuda", (
            "Found some offloaded weights. This is not compatible with reshaping "
            "experts to 3D prior model save. Ensure the model is fully on cuda."
        )
        m.to_3d_expert()
        print(f"Updated experts of {n}")

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
