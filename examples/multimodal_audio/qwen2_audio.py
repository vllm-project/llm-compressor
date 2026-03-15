from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# Select model and load it.
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "MLCommons/peoples_speech"
DATASET_SUBSET = "test"
DATASET_SPLIT = "test"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load raw dataset for generation testing.
raw_ds = load_dataset(
    DATASET_ID,
    DATASET_SUBSET,
    split=f"{DATASET_SPLIT}[:1]",
)

# Load dataset for calibration.
ds = load_dataset(
    DATASET_ID,
    DATASET_SUBSET,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
)


def preprocess(example):
    # Qwen2Audio uses a chat template format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "placeholder"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What did the person say?"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": example["text"]},
            ],
        },
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Process using the processor (it handles audio token expansion)
    inputs = processor(
        text=text,
        audio=[example["audio"]["array"]],
        sampling_rate=example["audio"]["sampling_rate"],
        return_tensors="pt",
    )

    # Strip batch dimension and return
    return {key: value[0] for key, value in inputs.items()}


ds = ds.map(preprocess, remove_columns=ds.column_names)

# Recipe
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head", "re:audio_tower.*"],
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the model before quantization.
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
raw_sample = raw_ds[0]
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio_url": "placeholder"},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What did the person say?"},
        ],
    },
]
text_prompt = processor.apply_chat_template(
    conversation, tokenize=False, add_generation_prompt=True
)
inputs = processor(
    text=text_prompt,
    audio=[raw_sample["audio"]["array"]],
    sampling_rate=raw_sample["audio"]["sampling_rate"],
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=100)
print(processor.batch_decode(output, skip_special_tokens=True)[0])
print("==========================================\n\n")
# that's where you have a lot of windows in the south no actually that's passive solar
# and passive solar is something that was developed and designed in the 1960s and 70s
# and it was a great thing for what it was at the time but it's not a passive house

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
