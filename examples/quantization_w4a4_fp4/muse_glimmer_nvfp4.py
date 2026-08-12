from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    MuseGlimmerForConditionalGeneration,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context

MODEL_ID = "meta-models/Muse-Glimmer-30B"

# Load model.
with load_context(MuseGlimmerForConditionalGeneration):
    model = MuseGlimmerForConditionalGeneration.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

DATASET_ID = "mlabonne/open-perfectblend"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}")
ds = ds.shuffle(seed=42)

ROLE_MAP = {"human": "user", "gpt": "assistant"}


def preprocess(example):
    messages = [
        {"role": ROLE_MAP.get(msg["from"], msg["from"]), "content": msg["value"]}
        for msg in example["conversations"]
    ]
    return {
        "text": processor.apply_chat_template(
            messages,
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return processor.tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm and scheme.
recipe = GPTQModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["re:.*vision.*", "lm_head", "re:.*embed_tokens.*"],
    offload_hessians=False,
    block_size=128,
)

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = processor.tokenizer(
    "Hello my name is", return_tensors="pt"
).input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(processor.tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-GPT-Shuffled"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
