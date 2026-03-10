"""
AWQ + GPTQModifier: Stacked Recipe Example
==========================================
Stacking AWQModifier with GPTQModifier combines AWQ's activation-aware
smoothing with GPTQ's second-order weight quantization for higher accuracy
at W4A16.

    recipe = [
        AWQModifier(...),
        GPTQModifier(...),
    ]

AWQModifier runs first and re-scales weights so that quantization-sensitive
channels become easier for GPTQ to handle.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import GPTQModifier

# ---------------------------------------------------------------------------
# 1. Model
# ---------------------------------------------------------------------------
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ---------------------------------------------------------------------------
# 2. Calibration dataset
# ---------------------------------------------------------------------------
DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


ds = ds.map(preprocess, remove_columns=ds.column_names)

# ---------------------------------------------------------------------------
# 3. Recipe: AWQ smoothing pass → GPTQ weight quantization
#
#   AWQModifier  : activation-aware smoothing (scale search uses scheme args).
#   GPTQModifier : Hessian-based weight quantization on the smoothed model.
#
#   Both modifiers must agree on scheme / targets / ignore.
# ---------------------------------------------------------------------------
recipe = [
    AWQModifier(
        ignore=["lm_head"],
        scheme="W4A16_ASYM",
        targets=["Linear"],
    ),
    GPTQModifier(
        scheme="W4A16_ASYM",
        targets=["Linear"],
        ignore=["lm_head"],
        dampening_frac=0.01,
    ),
]

# ---------------------------------------------------------------------------
# 4. Apply
# ---------------------------------------------------------------------------
OUTPUT_DIR = MODEL_ID.split("/")[-1] + "-AWQ-GPTQ-W4A16"

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=ds,
    recipe=recipe,
    output_dir=OUTPUT_DIR,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print(f"\nSaved to: {OUTPUT_DIR}")
