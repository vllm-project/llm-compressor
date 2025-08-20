from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridParallelExperts,
)
from compressed_tensors.utils import replace_module
from llmcompressor import oneshot
from llmcompressor.modeling.granite4 import GraniteMoeHybridParallelExpertsLinear
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation


MODEL_ID = "/net/storage149/autofs/css22/nmg/models/cos/f05940d/lake-models/models/base_training/shared/granite-4.0-small-base-prerelease-greylock-128k/r250709a"
# MODEL_ID = "ibm-granite/granite-4.0-tiny-preview"
# MODEL_ID = "/net/storage149/autofs/css22/cliu22/ssm_state_compression/gr4small_fp8_skipRouter_dequant"
# MODEL_ID = "gr4small_fp8_skipRouter_lin"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

skip_router_only = True
# In order to enable the quantization for Granite4 moe layers (.input_linear and .output_linear),
#   swap `GraniteMoeHybridParallelExperts` modules with `GraniteMoeHybridParallelExpertsLinear`.
#   This will convert moe expert tensors from 3D to 2D, i.e. [num_experts * out_feat, in_feat]
if skip_router_only:
    for n, m in model.named_modules():
        if isinstance(m, GraniteMoeHybridParallelExperts):
            new_mod = GraniteMoeHybridParallelExpertsLinear(
                m.num_experts, m.input_size, m.output_size,
            )
            new_mod.from_3d_expert(m)
            replace_module(model, n, new_mod)
            print(f"Replaced {n}")
    ignore_lay = ['re:.*lm_head', 're:.*block_sparse_moe.router']
    SAVE_DIR = "storage/gr4small_fp8_skipRouter_lin_exp3d_DEBUG"

# If skipping all .input_linear, .output-linear, and router layers, no need to swap.
else:
    ignore_lay = ['re:.*lm_head', 're:.*block_sparse_moe']
    SAVE_DIR = "storage/gr4small_fp8_skipMoe_lin_DEBUG"

recipe = QuantizationModifier(
    targets=["Linear", "GraniteMoeHybridParallelExpertsLinear"],
    scheme="FP8_DYNAMIC",
    ignore=ignore_lay)

# Apply quantization and save in compressed-tensors format.
# NOTE Conversion of weights from BF16 to FP8 will ONLY be triggered during model save. But if the
#       weights are converted, some layers may have F.Linear(bf16, fp8), which will not work
oneshot(model=model, recipe=recipe)  # , output_dir=SAVE_DIR

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

# NOTE During forward, weights in CompressedLinear will be decompressed -> compress again before .
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
