from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot

# NOTE: transformers 4.48.0 has an import error with DeepSeek.
# Please consider either downgrading your transformers version to a
# previous version or upgrading to a version where this bug is fixed

# select a Mixture of Experts model for quantization
MODEL_ID = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
# its recommended to use more calibration samples for MoE models so each expert is hit
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 2048
MAX_SEQUENCE_LENGTH = 2048


# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


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

# define a llmcompressor recipe for FP8 W8A8 quantization
# since the MoE gate layers are sensitive to quantization, we add them to the ignore
# list so they remain at full precision
recipe = [
    QuantizationModifier(
        targets="Linear",
        scheme="FP8",
        ignore=["lm_head", "re:.*mlp.gate$"],
    ),
]

SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8"

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    save_compressed=True,
    output_dir=SAVE_DIR,
)

print("========== SAMPLE GENERATION ==============")
SAMPLE_INPUT = ["I love quantization because"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
inputs = tokenizer(SAMPLE_INPUT, return_tensors="pt", padding=True).to(model.device)
output = model.generate(**inputs, max_length=50)
text_output = tokenizer.batch_decode(output)
print(text_output)
