# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
import torch
from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import preset_name_to_scheme
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen3-30B-A3B"

# Load the model with automatic device placement. Qwen3-30B-A3B fits on a
# single 80 GB GPU in bfloat16 for this calibration setup.
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Start from the NVFP4A16 preset (FP4 weights with group size 16), then use
# the expanded imatrix observer to choose ranges with activation importance.
scheme = preset_name_to_scheme("NVFP4A16", ["Linear"])
scheme.weights.observer = "nvfp4_expanded_imatrix"

recipe = [
    QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
    ),
]

# Qwen3-30B-A3B uses 32 calibration sequences of up to 512 tokens here.
oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset="perfectblend",
    splits="train[:32]",
    recipe=recipe,
    max_seq_length=512,
    num_calibration_samples=32,
    moe_calibrate_all_experts=True,
)

print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

save_dir = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4A16-expanded-imatrix"
model.save_pretrained(save_dir, save_compressed=True)
tokenizer.save_pretrained(save_dir)
