# NOTE: to use a custom dataset with your own data, see examples/custom_dataset_example.py
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context

MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"

with load_context(AutoModelForCausalLM):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = GPTQModifier(
    targets="Linear",
    scheme="FP8",
    ignore=[
        r"re:.*conv1d.*",
        r"backbone\.embeddings",
        r"re:.*_latent_proj.*",
        r"re:.*mixer.gate\..*",
        r"re:mtp.layers.*",
        "backbone.norm_f",
        "lm_head",
    ],
)

oneshot(
    model=model,
    dataset="perfectblend",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
