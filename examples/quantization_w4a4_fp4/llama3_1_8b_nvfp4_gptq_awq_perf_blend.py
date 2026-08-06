from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from compressed_tensors.quantization.quant_scheme import NVFP4

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

MODEL_ID = "RedHatAI/Llama-3.1-8B-Instruct"

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

DATASET_ID = "mlabonne/open-perfectblend"
DATASET_SPLIT = "train"

# Select number of samples. 512 samples is recommended for GPTQ.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

ROLE_MAP = {"human": "user", "gpt": "assistant"}


def preprocess(example):
    messages = [
        {"role": ROLE_MAP.get(msg["from"], msg["from"]), "content": msg["value"]}
        for msg in example["conversations"]
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
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

# Configure the quantization algorithm and scheme.
recipe = [
    AWQModifier(duo_scaling="both"),
    GPTQModifier(
        targets="Linear", 
        scheme="NVFP4", 
        ignore=["lm_head"],
    ),
]

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
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk in compressed-tensors format.
SAVE_DIR = "/data/dsikka/" + MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-GPTQ-AWQ-perf-blend"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
