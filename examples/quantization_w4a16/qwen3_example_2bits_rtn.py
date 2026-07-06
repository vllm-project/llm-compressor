from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Select model and load it.
model_id = "Qwen/Qwen3-8B"
# model_id = "/storage/yiliu7/Qwen/Qwen3-30B-A3B-Instruct-2507"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

ATTENTION_TARGETS = ["re:.*self_attn\\.(q|k|v)_proj$"]


# `W2A16` uses group_size=128 in the current compressed-tensors preset.
# This recipe only quantizes q/k/o attention projections and intentionally skips
# `v_proj`, which is noticeably more fragile at 2-bit on Qwen3.
# This is still experimental, but is materially better than quantizing every
# attention projection or every Linear layer to 2-bit with RTN.
recipe = QuantizationModifier(targets=ATTENTION_TARGETS, scheme="W2A16", ignore=["lm_head"])

# Apply algorithms.
oneshot(
    model=model,
    recipe=recipe,
)

# Sample generation. This skip-`v_proj` variant is a better 2-bit RTN stress
# test than full attention or full-model W2A16, but it is still not a general
# quality baseline.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
print("[note] This recipe quantizes q/k/o only and keeps v_proj dense.")
dispatch_model(model)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Introduce yourself in one sentence."}],
    tokenize=False,
    add_generation_prompt=True,
)
sample = tokenizer(prompt, return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# # Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W2A16-G128-AttnQKV-RTN"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
