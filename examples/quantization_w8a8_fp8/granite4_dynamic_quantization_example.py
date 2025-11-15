import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modeling import replace_modules_for_calibration

MODEL_ID = "ibm-granite/granite-4.0-h-small"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = replace_modules_for_calibration(model)

ignore_lay = ["lm_head"]

recipe = QuantizationModifier(
    targets=["Linear"],
    scheme="FP8_DYNAMIC",
    ignore=ignore_lay,
)

oneshot(model=model, recipe=recipe)

print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
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
