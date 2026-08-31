# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context

# select a Mixture of Experts model for quantization
model_id = "ibm-research/PowerMoE-3b"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# define a llmcompressor recipe for W416 quantization with a group size of 128
# since the MoE gate layers are sensitive to quantization, we add them to the ignore
# list so they remain at full precision

recipe = GPTQModifier(
    targets=["Linear"],
    scheme="NVFP4",
    ignore=["lm_head", "re:.*.block_sparse_moe.router.layer"],
)

oneshot(
    model=model,
    dataset="perfectblend",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
    save_compressed=True,
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-FP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
