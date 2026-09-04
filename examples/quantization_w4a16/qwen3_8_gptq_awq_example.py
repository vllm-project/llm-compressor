# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen3.8-27B"

# Load model.
with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)


recipe = [
    AWQModifier(duo_scaling="both"),
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
            "re:visual.*",
            "re:model.visual.*",
            r"re:.*lm_head",
            "re:.*embed_tokens$",
            r"re:.*linear_attn\.in_proj_a$",
            r"re:.*linear_attn\.in_proj_b$",
        ],
    ),
]

# Apply quantization.
oneshot(
    model=model,
    processor=processor,
    recipe=recipe,
    dataset="perfectblend",
    splits="train[:512]",
    max_seq_length=4096,
    num_calibration_samples=512,
    moe_calibrate_all_experts=True,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-GPTQ-AWQ"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
