from pathlib import Path

from auto_round.calib_dataset import get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from compressed_tensors.offload import dispatch_model

# Select model and load it.
model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048
# Get aligned calibration dataset.

ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)

# Configure the quantization algorithm to run.
#   * quantize the model to W4A4-MXFP4 with AutoRound
recipe = AutoRoundModifier(
    targets="Linear",
    scheme="MXFP4",
    ignore=["lm_head"],
    iters=200,
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    # disable shuffling to get slightly better mmlu score
    shuffle_calibration_samples=False,
)

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
SAVE_DIR = Path(model_id).name + "-W4A4-MXFP4-AutoRound"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
