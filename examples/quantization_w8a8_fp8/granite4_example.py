from compressed_tensors.utils import replace_module
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridParallelExperts,
)

from llmcompressor import oneshot
from llmcompressor.modeling.granite4 import GraniteMoeHybridParallelExpertsLinear
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

"""Please see details in `README_granite4.md`."""

MODEL_ID = "ibm-granite/granite-4.0-tiny-preview"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

skip_router_only = True  # assume we want to quantize input/output moe layers
ignore_lay = [
    "lm_head",
]
if skip_router_only:
    # swap moe linears to a custom class
    for n, m in model.named_modules():
        if isinstance(m, GraniteMoeHybridParallelExperts):
            new_mod = GraniteMoeHybridParallelExpertsLinear.from_3d_expert(m)
            replace_module(model, n, new_mod)
    ignore_lay += ["re:.*block_sparse_moe.router"]
    SAVE_DIR = "ibm-granite-4-tiny-fp8-dynamic-skipMoeRouter"
else:
    # Skip all .input_linear, .output-linear, and router layers.
    ignore_lay += ["re:.*block_sparse_moe"]
    SAVE_DIR = "ibm-granite-4-tiny-fp8-dynamic-skipMoe"

recipe = QuantizationModifier(
    targets=["Linear", "GraniteMoeHybridParallelExpertsLinear"],
    scheme="FP8_DYNAMIC",
    ignore=ignore_lay,
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer(
    "What is your favorite TV show?", return_tensors="pt"
).input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# Revert weights of MoE experts to 3D format (num_experts, output_size, input_size)
for n, m in model.named_modules():
    if isinstance(m, GraniteMoeHybridParallelExpertsLinear):
        # NOTE: can assert type != "meta" instead, which is sign of offloading
        assert m.weight.device.type == "cuda", (
            "Found some offloaded weights. This is not compatible with reshaping "
            "experts to 3D prior model save. Ensure the model is fully on cuda."
        )
        m.to_3d_expert()

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
