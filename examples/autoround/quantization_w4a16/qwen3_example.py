from auto_round.calib_dataset import get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
model_id = "Qwen/Qwen3-235B-A22B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048
ITERS = 200
# Get aligned calibration dataset.

ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)


# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with AutoRound with a group size 128
#   * For `Qwen/Qwen3-235B-A22B`, it requires about 300 GB memory
#     to run tuning with default settings.
recipe = AutoRoundModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*mlp.gate$",
    ],
    iters=ITERS,
    enable_torch_compile=False,
    device_ids="0,1,2,3",  # Use 4 A100 GPUs
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
dispatch_for_generation(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128-AutoRound"
print(f"save to {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
