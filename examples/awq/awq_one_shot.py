import lm_eval
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from lm_eval.utils import make_table
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

# This example demonstrates how to:
# 1) Run the `llm-compressor` implementation of AWQ
# 2) Evaluate the compressed model with the lm_eval framework

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_ID = "mit-han-lab/pile-val-backup"
DATASET_SPLIT = "validation"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512
OUTPUT_DIR = MODEL_ID.split("/")[-1] + "-awq-asym"

#
# 1) Run LLM Compressor AWQ implementation
#

recipe = [
    AWQModifier(bits=4, symmetric=False),
    QuantizationModifier(
        ignore=["lm_head"],
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type=QuantizationType.INT,
                    dynamic=False,
                    symmetric=False,
                    strategy=QuantizationStrategy.GROUP,
                    group_size=128,
                ),
            )
        },
    ),
]

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def get_calib_dataset(tokenizer):
    from datasets import load_dataset

    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES*100}]",
    )

    def preprocess(example):
        return {
            "input_ids": tokenizer.encode(example["text"].strip())[:MAX_SEQUENCE_LENGTH]
        }

    ds = (
        ds.shuffle(seed=42)
        .map(preprocess, remove_columns=ds.column_names)
        .filter(lambda example: len(example["input_ids"]) >= MAX_SEQUENCE_LENGTH)
        .select(range(NUM_CALIBRATION_SAMPLES))
    )

    return ds


oneshot(
    model=model,
    dataset=get_calib_dataset(tokenizer=tokenizer),
    recipe=recipe,
    output_dir=OUTPUT_DIR,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("Done! model saved to", OUTPUT_DIR)

#
# 2) Evaluate model on wikitext perplexity
#

results = lm_eval.simple_evaluate(
    model="vllm",
    model_args={
        "pretrained": OUTPUT_DIR,
        "add_bos_token": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.5,
    },
    tasks=["wikitext"],
    num_fewshot=5,
    batch_size="auto",
)
print(make_table(results))
