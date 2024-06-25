import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot

model_stub = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
output_dir = "/cache/llm-compressor/tiny_fp8_test"
num_calibration_samples = 512

tokenizer = AutoTokenizer.from_pretrained(model_stub, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


def preprocess(batch):
    text = tokenizer.apply_chat_template(batch["messages"], tokenize=False)
    tokenized = tokenizer(text, padding=True, truncation=True, max_length=2048)
    return tokenized


ds = load_dataset("mgoin/ultrachat_2k", split="train_sft")
examples = ds.map(preprocess, remove_columns=ds.column_names)

recipe = QuantizationModifier(targets="Linear", scheme="FP8")

model = SparseAutoModelForCausalLM.from_pretrained(
    model_stub, torch_dtype=torch.bfloat16, device_map="auto"
)

oneshot(
    model=model,
    dataset=examples,
    recipe=recipe,
    output_dir=output_dir,
    num_calibration_samples=num_calibration_samples,
    save_compressed=True,
)
