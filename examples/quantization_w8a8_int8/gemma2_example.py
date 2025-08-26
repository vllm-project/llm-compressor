from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

# 1) Select model and load it.
MODEL_ID = "google/gemma-2-2b-it"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2) Prepare calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# 3) Select quantization algorithms. In this case, we:
#   * quantize the weights to int8 with GPTQ (static per channel)
#   * quantize the activations to int8 (dynamic per token)
recipe = GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])

# 4) Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
# NOTE: transformers 4.49.0 results in a generation error with gemma2.
# Consider either downgrading your transformers version to a previous version
# or use vLLM for sample generation.
# Note: compile is disabled: https://github.com/huggingface/transformers/issues/38333
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20, disable_compile=True)
print(tokenizer.decode(output[0]))
print("==========================================")

# 5) Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-INT8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
