import torch
from datasets import load_dataset

from llmcompressor.transformers import (
    SparseAutoModelForCausalLM,
    SparseAutoTokenizer,
    oneshot,
)

# define a llmcompressor recipe for GPTQ W4A16 quantization
recipe = """
quant_stage:
    quant_modifiers:
        GPTQModifier:
            sequential_update: false
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 4
                        type: "int"
                        symmetric: true
                        strategy: "channel"
                    targets: ["Linear"]
"""

# load in a 50% sparse model with 2:4 sparsity structure
# setting device_map to auto to spread the model evenly across all available GPUs
model_stub = "neuralmagic/SparseLlama-2-7b-cnn-daily-mail-pruned_50.2of4"
model = SparseAutoModelForCausalLM.from_pretrained(
    model_stub, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = SparseAutoTokenizer.from_pretrained(model_stub)

# for quantization calibration, we will use a subset of the dataset that was used to
# sparsify and finetune the model
dataset = load_dataset("abisee/cnn_dailymail", "1.0.0", split="train[:5%]")

# set dataset config parameters
max_seq_length = 4096
pad_to_max_length = False
num_calibration_samples = 1024


# preprocess the data into a single text entry, then tokenize the dataset
def process_sample(sample):
    formatted = "Article:\n{}\n\n### Summarization:\n{}".format(
        sample["article"], sample["highlights"]
    )
    return tokenizer(
        formatted, padding=pad_to_max_length, max_length=max_seq_length, truncation=True
    )


tokenized_dataset = dataset.map(
    process_sample, remove_columns=["article", "highlights", "id"]
)

breakpoint()
# save location of quantized model out
output_dir = "./llama7b_sparse_24_w4a16_channel_compressed"

# apply quantization recipe to the model and save quantized output int4 packed format
# the sparsity structure of the original model will be maintained
oneshot(
    model=model,
    dataset=tokenized_dataset,
    recipe=recipe,
    output_dir=output_dir,
    max_seq_length=max_seq_length,
    pad_to_max_length=pad_to_max_length,
    num_calibration_samples=num_calibration_samples,
    save_compressed=True,
)
