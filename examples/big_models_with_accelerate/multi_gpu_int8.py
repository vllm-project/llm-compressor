from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot

MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
SAVE_DIR = MODEL_ID.split("/")[1] + "-W8A8-Dynamic"

# 1) Load model (device_map="auto" with shard the model over multiple GPUs!).
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2) Prepare calibration dataset (in this case, we use ultrachat).
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
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

# 3) Configure algorithms. In this case, we:
#   * quantize the weights to int8 with GPTQ (static per channel)
#   * quantize the activations to int8 (dynamic per token)
recipe = [
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# 4) Apply algorithms and save in `compressed-tensors` format.
# if you encounter GPU out-of-memory issues, consider using an explicit
# device map (see multi_gpus_int8_device_map.py)
oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    output_dir=SAVE_DIR,
)
