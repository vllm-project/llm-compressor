# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

# Step3p5 uses custom modeling code from the Hub.
MODEL_ID = "stepfun-ai/Step-3.5-Flash"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Configure the quantization algorithm to run.
# The Step3p5 AWQ mapping is selected automatically from the model architecture.
# Keep the MoE router in higher precision because Step3p5 routes with fp32 gates.
recipe = [
    AWQModifier(),
    QuantizationModifier(
        ignore=["lm_head", "re:.*moe.gate$"],
        scheme="W4A16",
        targets=["Linear"],
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=512,
    num_calibration_samples=256,
    trust_remote_code_model=True,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq-sym"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
