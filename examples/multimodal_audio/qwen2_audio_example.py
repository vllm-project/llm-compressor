from datasets import load_dataset
from transformers import AutoProcessor

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.tracing import (
    TraceableQwen2AudioForConditionalGeneration,
)
from llmcompressor.transformers.utils.data_collator import qwen2_audio_data_collator

# Select model and load it.
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

model = TraceableQwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "MLCommons/peoples_speech"
DATASET_SUBSET = "test"
DATASET_SPLIT = "test"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(
    DATASET_ID,
    DATASET_SUBSET,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
    trust_remote_code=True,
)


def preprocess(example):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": None},
                {"type": "text", "text": "What does the person say?"},
            ],
        },
    ]

    return {
        "text": processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        ),
        "audios": [example["audio"]["array"]],
        "sampling_rate": example["audio"]["sampling_rate"],
    }


ds = ds.map(preprocess, remove_columns=ds.column_names)


# Tokenize inputs.
def tokenize(sample):
    return processor(**sample, return_tensors="pt")


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with GPTQ with a group size 128
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "re:audio_tower.*",
        "re:multi_modal_projector.*",
        "lm_head",
    ],  # TODO: honestly, there's a decent number of parameters in the audio tower worth quantizing
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=qwen2_audio_data_collator,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
breakpoint()
sample_input = qwen2_audio_data_collator([next(iter(ds))])
sample_input = {k: v.to(model.device) for k, v in sample_input.items()}
output = model.generate(**sample_input)
print(processor.batch_decode(output, skip_special_tokens=True)[0])
print("==========================================\n\n")
# that's where you have a lot of windows in the

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
