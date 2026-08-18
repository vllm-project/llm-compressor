from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from compressed_tensors.distributed import init_dist

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context

# Select model and load it.
init_dist()
model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")

# Select calibration dataset.
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

# Prune 25% of the experts in each MoE layer, based on saliency.
# You can adjust this value to prune more or less aggressively.
recipe = [
    REAPPruningModifier(sparsity=0.25),
    GPTQModifier(
        targets=r"re:.*mlp.*",
        scheme="NVFP4",
        ignore=["re:.*mlp.gate$", "re:.*shared_expert_gate.*"]
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=False,  # Disable calibrating all experts for REAP
    pipeline="sequential",
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-NVFP4-REAP-25"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
