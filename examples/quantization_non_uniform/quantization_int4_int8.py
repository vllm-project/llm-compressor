# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize all weights excluding down_proj layers
#       to int4 with per group 128 via ptq
#   * quantize all down_proj layer weights to int8
#       with per group 128 via ptq
recipe = """
quant_stage:
    quant_modifiers:
        GPTQModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: int
                        strategy: group
                        dynamic: false
                        symmetric: true
                        group_size: 128
                    targets: ["re:.*down_proj.*"]
                group_1:
                    weights:
                        num_bits: 4
                        type: int
                        strategy: group
                        dynamic: false
                        symmetric: false
                        group_size: 128
                    targets: ["re:.*self_attn.k_proj.*", "re:.*self_attn.o_proj.*",
                        "re:.*self_attn.q_proj.*", "re:.*self_attn.v_proj.*",
                        "re:.*gate_proj.*", "re:.*up_proj.*"]
"""
# Apply quantization.
oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")


# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-W8A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
