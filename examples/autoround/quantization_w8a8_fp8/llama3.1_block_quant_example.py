from auto_round.calib_dataset import get_dataset
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

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
# NOTE: AutoRoundModifier with iters=0 is equivalent to RTN
recipe = AutoRoundModifier(
    targets="Linear", scheme="FP8_BLOCK", ignore=["lm_head"], iters=200
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
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK-AutoRound"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
