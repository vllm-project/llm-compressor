from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.quant_args import DynamicType, FP8_E4M3_DATA
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 20
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

# Configure the quantization algorithm and scheme.
# Uses Four Over Six (4/6) adaptive block scaling for NVFP4 weights:
#   * For each block, quantizes with both M=6 and M=4, selects the scale
#     with lower quantization error (MSE by default).
#   * Uses a modified global scale (M_FP8=256 instead of 448) so all blocks
#     can benefit from M=4 scaling.
#   * Activations use standard NVFP4 dynamic quantization.
#
# Scale selection options (via observer_kwargs["scale_selection"]):
#   "mse"     - mean squared error (default, best for PTQ)
#   "mae"     - mean absolute error (best for pre-training)
#   "abs_max" - maximum absolute error
#
# Reference: Cook et al., "Four Over Six: More Accurate NVFP4 Quantization
# with Adaptive Block Scaling", arXiv:2512.02010, 2025.

recipe = QuantizationModifier(
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                type=QuantizationType.FLOAT,
                strategy=QuantizationStrategy.TENSOR_GROUP,
                symmetric=True,
                dynamic=False,
                group_size=16,
                scale_dtype=FP8_E4M3_DATA.dtype,
                zp_dtype=FP8_E4M3_DATA.dtype,
                observer="fouroversix",
                observer_kwargs={"scale_selection": "mse"},
            ),
            input_activations=QuantizationArgs(
                num_bits=4,
                type=QuantizationType.FLOAT,
                strategy=QuantizationStrategy.TENSOR_GROUP,
                symmetric=True,
                dynamic=DynamicType.LOCAL,
                group_size=16,
                observer="static_minmax",
                scale_dtype=FP8_E4M3_DATA.dtype,
                zp_dtype=FP8_E4M3_DATA.dtype,
            ),
        ),
    },
    ignore=["lm_head"],
)

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-FourOverSix"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
