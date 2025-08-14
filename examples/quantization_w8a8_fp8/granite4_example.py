from transformers import AutoModelForCausalLM, AutoTokenizer

from compressed_tensors.utils import replace_module
from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modeling.granite4 import (
    GraniteMoeHybridParallelExperts,
    GraniteMoeHybridParallelExpertsModList,
)
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "/net/storage149/autofs/css22/nmg/models/cos/f05940d/lake-models/models/base_training/shared/granite-4.0-small-base-prerelease-greylock-128k/r250709a"
# MODEL_ID = "ibm-granite/granite-4.0-tiny-preview"
# MODEL_ID = "/net/storage149/autofs/css22/cliu22/ssm_state_compression/gr4small_fp8_skipRouter_dequant"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Confirm generations of the original model look sane.
print("Before module swapping")
print(model)
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
# input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to('cuda')
input_ids = tokenizer("What is your favorite TV show?", return_tensors="pt").input_ids.to('cuda')
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")


skip_router_only = False

# In order to enable the quantization for Granite4 moe layers (.input_linear and .output_linear),
#   swap `GraniteMoeHybridParallelExperts` modules with `GraniteMoeHybridParallelExpertsModList`.
#   This will break moe expert tensors from 3D to N*2D, i.e. [num_experts, out_feat, in_feat] into
#   (num_experts x nn.Linear), where Linear's weight is [out_feat, in_feat]
if skip_router_only:
    model = replace_modules_for_calibration(model)
    ignore_lay = ['re:.*lm_head', 're:.*block_sparse_moe.router']
    SAVE_DIR = "gr4small_fp8_skipRouter"

# If skipping all .input_linear, .output-linear, and router layers, no need to break and swap.
else:
    ignore_lay = ['re:.*lm_head', 're:.*block_sparse_moe']
    SAVE_DIR = "gr4small_fp8_skipMoe"

recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=ignore_lay)

# Apply quantization and save in compressed-tensors format.
# NOTE Conversion of weights from BF16 to FP8 will ONLY be triggered during model save. But if the
#       weights are converted, some layers may have F.Linear(bf16, fp8), which will not work
oneshot(model=model, recipe=recipe)  # , output_dir=SAVE_DIR

# Confirm generations of the quantized model look sane.
print("After module swapping")
print(model)
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
# input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
input_ids = tokenizer("What is your favorite TV show?", return_tensors="pt").input_ids.to('cuda')
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# NOTE Before correct implementation in HF or VLLM, if we want to verify model accuracy, we may
#       choose to dequantize the weights, concat back to 3D, and revert the modules back to 
#       `GraniteMoeHybridParallelExperts`.
save_dequant_3d = False
if save_dequant_3d:
    SAVE_DIR += "_dequant"
    for n, m in model.named_modules():
        if isinstance(m, GraniteMoeHybridParallelExpertsModList):
            new_mod = GraniteMoeHybridParallelExperts(m.num_experts, m.input_size, m.output_size)
            new_mod.weight.data = m.dequant_experts_3d_weight()
            new_mod.weight.requires_grad = False
            replace_module(model, n, new_mod)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)



