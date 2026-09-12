"""Smoke test for QuantizationAwareDistillationModifier."""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.distillation import QuantizationAwareDistillationModifier

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

NUM_CALIBRATION_SAMPLES = 32
MAX_SEQUENCE_LENGTH = 512

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:64]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        )
    }


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

recipe = QuantizationAwareDistillationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    epochs=2,
    lr=1e-5,
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
print("QAD smoke test passed!")
