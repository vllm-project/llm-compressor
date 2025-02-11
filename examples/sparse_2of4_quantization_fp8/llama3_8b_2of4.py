import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import post_train
from llmcompressor.modifiers.obcq import SparseGPTModifier
from llmcompressor.modifiers.pruning import ConstantPruningModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configuration
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Apply compression to a model")
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 compression")
    return parser.parse_args()


def preprocess(example):
    """Preprocess dataset examples."""
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


def tokenize(sample):
    """Tokenize dataset examples."""
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


def get_recipe(fp8_enabled):
    """Generate the compression recipe and save directory based on the FP8 flag."""
    base_recipe = [
        SparseGPTModifier(
            sparsity=0.5,
            mask_structure="2:4",
            sequential_update=True,
            targets=[r"re:model.layers.\d*$"],
        )
    ]
    save_dir = MODEL_ID.split("/")[1] + "2of4-sparse"

    if fp8_enabled:
        base_recipe.extend(
            [
                QuantizationModifier(
                    targets=["Linear"],
                    ignore=["lm_head"],
                    scheme="FP8_DYNAMIC",
                ),
                ConstantPruningModifier(
                    targets=[
                        r"re:.*q_proj.weight",
                        r"re:.*k_proj.weight",
                        r"re:.*v_proj.weight",
                        r"re:.*o_proj.weight",
                        r"re:.*gate_proj.weight",
                        r"re:.*up_proj.weight",
                        r"re:.*down_proj.weight",
                    ],
                    start=0,
                ),
            ]
        )
        save_dir = MODEL_ID.split("/")[1] + "2of4-W8A8-FP8-Dynamic-Per-Token"

    return base_recipe, save_dir


# Parse arguments
args = parse_args()

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load and preprocess dataset
ds = (
    load_dataset(DATASET_ID, split=DATASET_SPLIT)
    .shuffle(seed=42)
    .select(range(NUM_CALIBRATION_SAMPLES))
)
ds = ds.map(preprocess)
ds = ds.map(tokenize, remove_columns=ds.column_names)

# Get compression recipe and save directory
recipe, save_dir = get_recipe(args.fp8)

# Apply compression
post_train(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Validate the compressed model
print("\n========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n")

# Save compressed model and tokenizer
model.save_pretrained(
    save_dir, save_compressed=args.fp8, disable_sparse_compression=True
)
tokenizer.save_pretrained(save_dir)
