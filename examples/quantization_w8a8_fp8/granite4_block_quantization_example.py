import torch
import torch.nn as nn
import torch.nn.functional as F

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modeling.granite4 import replace_granite_moe_with_linear_experts, pack_3d_experts

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ibm-granite/granite-4.0-h-small"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = replace_granite_moe_with_linear_experts(model)

ignore_lay = ["lm_head"]
ignore_lay += ["re:.*block_sparse_moe.router"]
ignore_lay += ["re:.*mamba.in_proj"]
ignore_lay += ["re:.*shared_mlp.input_linear"]

recipe = QuantizationModifier(
    targets=["Linear"],
    scheme="FP8_BLOCK",
    ignore=ignore_lay,
)

oneshot(model=model, recipe=recipe)
dispatch_for_generation(model)

print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer(
    "Describe Large Language Model", return_tensors="pt"
).input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=35)
print(tokenizer.decode(output[0]))
print("==========================================")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-block"
print(f"Saving to {SAVE_DIR}")

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

pack_3d_experts(SAVE_DIR)
