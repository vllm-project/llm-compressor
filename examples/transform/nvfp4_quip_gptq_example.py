"""
NOTE: On Blackwell consumer GPUs (RTX 5090 / SM120), models produced by this
example require vLLM with the hadacore_transform fix:
https://github.com/vllm-project/vllm/pull/43462
"""

from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import (
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform import QuIPModifier

# Select model and load it.
model_id = "meta-llama/Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
# WikiText-2 is the standard calibration corpus in quantization research
# (QuIP, GPTQ, AutoRound, MR-GPTQ arXiv:2509.23202), enabling direct comparison
# against published PPL numbers on this composition.
DATASET_ID = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "train"

# Select number of samples. 256 samples of 2048 tokens is sufficient for
# 4-bit quantization on Llama-3.1-8B.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and concatenate into fixed-length chunks. WikiText-2 articles
# are much shorter than MAX_SEQUENCE_LENGTH, so per-article tokenization would
# yield many short sequences and waste calibration capacity.
raw_ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT)
full_text = "\n\n".join(t for t in raw_ds["text"] if t.strip())
input_ids_all = tokenizer(
    full_text, return_tensors="pt", add_special_tokens=False
).input_ids[0]
n_chunks = min(NUM_CALIBRATION_SAMPLES, input_ids_all.size(0) // MAX_SEQUENCE_LENGTH)
ds = Dataset.from_list(
    [
        {
            "input_ids": input_ids_all[
                i * MAX_SEQUENCE_LENGTH : (i + 1) * MAX_SEQUENCE_LENGTH
            ].tolist()
        }
        for i in range(n_chunks)
    ]
)

# Configure the quantization algorithm to run.
#   * apply quip transforms to model in order to make quantization easier
#   * quantize the weights to nvfp4 (fp4 weights, fp16 activations) with a
#     hardware-imposed group size of 16
#   * use GPTQ with mse observer for activation-ordering-aware calibration
#     (combination follows the MR-GPTQ recipe, arXiv:2509.23202)
NVFP4A16 = dict(
    weights=QuantizationArgs(
        num_bits=4,
        actorder="static",
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        symmetric=True,
        dynamic=False,
        group_size=16,
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=FP8_E4M3_DATA.dtype,
        observer="mse",
    ),
    targets=["Linear"],
)
recipe = [
    QuIPModifier(
        rotations=["v", "u"], transform_block_size=128, transform_type="hadamard"
    ),
    GPTQModifier(config_groups={"group_0": NVFP4A16}, ignore=["lm_head"]),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-quip-gptq-nvfp4a16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
