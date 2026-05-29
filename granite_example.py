# 1. Adjust recipe and preprocess
# 2. Quantized models in vLLM standardize on 2d experts. vLLM will fused into 3d weights for efficiency at load time
# 3. Ask claude to compare unquantized and quantized checkpoints and fix them if different

import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modeling.moe.linearize import load_quantizable_moe

# select a Mixture of Experts model for quantization
MODEL_ID = "ibm-research/PowerMoE-3b"

with load_quantizable_moe():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048  # longer is better


# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


def preprocess(example):
    return {
        "text": str(example["messages"])  # TODO: apply a real chat template. Or don't.
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

# define a llmcompressor recipe for W416 quantization with a group size of 128
# since the MoE gate layers are sensitive to quantization, we add them to the ignore
# list so they remain at full precision

# QuantizationModifier := round to nearest
# GPTQModifier := more advanced

recipe = GPTQModifier(  # TODO: consider using QuantizationModifier as first pass
    targets=["Linear"],
    scheme="FP8",  # TODO: change to NVFP4
    ignore=[
        "lm_head",
        "re:.*.block_sparse_moe.router.layer"
    ] + [f"re:model.layers.{i}.*" for i in range(1, 100)],  # TODO: remove
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    save_compressed=True,
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
