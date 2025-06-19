import torch
from datasets import load_dataset
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

# NOTE: transformers 4.49.0 has an attribute error with DeepSeek.
# Please consider either downgrading your transformers version to a
# previous version or upgrading to a version where this bug is fixed

# select a Mixture of Experts model for quantization
MODEL_ID = "deepseek-ai/DeepSeek-V2.5"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
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

# Configure the quantization algorithm to run.
# since the MoE gate layers are sensitive to quantization, we add them to the ignore
# list so they remain at full precision
recipe = GPTQModifier(
    targets="Linear", scheme="W4A16", ignore=["lm_head", "re:.*mlp.gate$"]
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
)

# Confirm generations of the quantized model look sane.
# Generation is broken for deepseek models when using the latest transformers package
if Version(__version__) < Version("4.48"):
    print("========== SAMPLE GENERATION ==============")
    dispatch_for_generation(model)
    sample = tokenizer("Hello my name is", return_tensors="pt")
    sample = {key: value.to("cuda") for key, value in sample.items()}
    output = model.generate(**sample, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================")
else:
    print(
        "WARNING: cannot perform sample generation of "
        "deepseek models with transformers >= 4.48"
    )

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
