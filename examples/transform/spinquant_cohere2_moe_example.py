"""
Apply SpinQuant R1+R2+R4 and NVFP4 (W4A4) quantization to
CohereLabs/North-Mini-Code-1.0, then save the compressed checkpoint.

North-Mini-Code-1.0 is a `Cohere2MoeForCausalLM` that the generic SpinQuant
pipeline can't prepare, so we call `prepare_cohere2_moe_for_spinquant` first.

NVFP4 normally calibrates a static per-tensor `input_global_scale`. For
simplicity this example uses `pipeline="datafree"` and skips that calibration,
setting the scale to 1.0 and relying on the runtime per-group fp8 scales. We
found this gives reasonable output -- this script is just a showcase, 
not a recipe for best accuracy.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from compressed_tensors.offload import update_offload_parameter
from llmcompressor import oneshot
from llmcompressor.modeling.moe.cohere2_moe import prepare_cohere2_moe_for_spinquant
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier

MODEL_ID = "CohereLabs/North-Mini-Code-1.0"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Required architecture-specific prep; MUST run before `oneshot`.
prepare_cohere2_moe_for_spinquant(model)

# block_size=128 divides hidden_size (2048), head_dim (128), and both MLP
# intermediate sizes (3072 dense, 768 expert), so "hadamard" works for all.
recipe = [
    SpinQuantModifier(
        rotations=["R1", "R2", "R4"],
        transform_block_size=128,
        transform_type="hadamard",
    ),
    QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            "lm_head",
            # keep attention projections and the MoE router in full precision
            r"re:.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
            r"re:.*mlp\.gate\.linear$",
        ],
    ),
]

oneshot(model=model, recipe=recipe, pipeline="datafree")

# Set the uncalibrated `input_global_scale` to 1.0 (see module docstring).
with torch.no_grad():
    for module in model.modules():
        global_scale = getattr(module, "input_global_scale", None)
        if global_scale is not None:
            update_offload_parameter(
                module, "input_global_scale", torch.ones_like(global_scale)
            )

# Sample generation skipped: R4 online rotation is unavailable in HF transformers.

SAVE_DIR = MODEL_ID.split("/")[1] + "-spinquantR1R2R4-nvfp4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Saved compressed model to {SAVE_DIR}")
