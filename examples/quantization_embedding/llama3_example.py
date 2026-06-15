from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Select model and load it.
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure the quantization algorithm to run.
#   * quantize the input embedding table to int4 with a group size of 64
#   * weight-only: there are no input/output activations to quantize for a lookup
#
# The target is the `Embedding` module *class* (not a name regex). Matching by
# class name keeps the recipe portable across architectures and independent of a
# model's module prefix, so the same recipe works whether the embedding is named
# `model.embed_tokens`, `gpt_neox.embed_in`, etc.
recipe = QuantizationModifier(
    config_groups={
        "embedding": {
            "targets": ["Embedding"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 64,
            },
        }
    }
)

# Apply the quantization algorithm. Embedding quantization is weight-only, so it
# is data-free -- no calibration dataset is required.
oneshot(model=model, recipe=recipe)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-embedding-W4A16-G64"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
