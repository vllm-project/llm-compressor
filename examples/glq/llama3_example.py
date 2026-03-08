from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.glq import GLQModifier

# Select model and load
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset
DATASET_ID = "HuggingFaceFW/fineweb-edu-score-2"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            [{"role": "user", "content": example["text"]}],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)

# Configure GLQ quantization
# bits=2: 16-bit index per 8 weights (2 bpw)
# bits=3: 16-bit + 8-bit residual index per 8 weights (3 bpw)
# bits=4: 16-bit + 16-bit residual index per 8 weights (4 bpw)
recipe = [GLQModifier(bits=2, ignore=["lm_head"])]

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    text_column="text",
)

# Save compressed model
SAVE_DIR = model_id.split("/")[-1] + "-GLQ-2bpw"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model saved to {SAVE_DIR}")
