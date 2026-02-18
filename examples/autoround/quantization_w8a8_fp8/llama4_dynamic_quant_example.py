from auto_round.calib_dataset import get_dataset
from compressed_tensors.offload import dispatch_model
from transformers import AutoProcessor, Llama4ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier

# Select model and load it.
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
model = Llama4ForConditionalGeneration.from_pretrained(model_id, dtype="auto")
processor = AutoProcessor.from_pretrained(model_id)

# Select calibration dataset.
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048
# Get aligned calibration dataset.

ds = get_dataset(
    tokenizer=processor.tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)


# Configure the quantization algorithm to run.
recipe = AutoRoundModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "re:.*lm_head",
        "re:.*router",
        "re:.*self_attn.*",
        "re:.*shared_expert.*",
        "re:multi_modal_projector.*",
        "re:vision_model",
    ],
    iters=0,
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
sample = processor(text="Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=1)
print(processor.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W8A8-Dynamic-AutoRound"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
