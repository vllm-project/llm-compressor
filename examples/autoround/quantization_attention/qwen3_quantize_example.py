from auto_round.calib_dataset import get_dataset
from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 1024
ITERS = 200

# Get aligned calibration dataset.
ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)

recipe = [
    QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=["Qwen3Attention"],
                input_activations=QuantizationArgs(
                    num_bits=8, type="float", strategy="tensor"
                ),
            ),
        },
    ),
    AutoRoundModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
        iters=ITERS,
    ),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    shuffle_calibration_samples=False,
)

print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=50)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-FP8Attention-W4A16-AutoRound"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print("Saved to", SAVE_DIR)
