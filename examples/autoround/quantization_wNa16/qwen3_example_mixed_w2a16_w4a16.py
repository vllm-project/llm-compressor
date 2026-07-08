from auto_round.calib_dataset import get_dataset
from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_scheme import W2A16, W4A16
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier

# Select model and load it.
model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048
ITERS = 200

ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)

# Configure the quantization algorithm to run.
#   * quantize attention weights to 2 bit with AutoRound with a group size 128
#   * quantize MLP weights to 4 bit with AutoRound with a group size 128
#   * keep lm_head dense
recipe = AutoRoundModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=[
                "re:.*self_attn\\.(q|k|v|o)_proj$",
            ],
            **W2A16,
        ),
        "mlp": QuantizationScheme(
            targets=[
                "re:.*mlp\\.(gate|up|down)_proj$",
            ],
            **W4A16,
        ),
    },
    ignore=["lm_head"],
    iters=ITERS,
    enable_torch_compile=False,
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    shuffle_calibration_samples=False,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
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

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-AttnW2A16-MlpW4A16-G128-AutoRound"
print(f"save to {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
