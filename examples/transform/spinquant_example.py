from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "mit-han-lab/pile-val-backup"
DATASET_SPLIT = "validation"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            [{"role": "user", "content": example["text"]}],
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


# NOTE: currently only fused rotations (R1 & R2) are available
# Learned rotations and online rotations (R3 & R4) will be added
# in a future release.
# Configure the quantization algorithm to run.
#   * apply spinquant transforms to model to reduce quantization loss
#   * quantize the weights to 4 bit with group size 128
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.quant_scheme import FP8

config_groups = {
    # "attention": QuantizationScheme(
    #     targets=["LlamaAttention"],
    #     input_activations=QuantizationArgs(
    #         num_bits=8,
    #         type=QuantizationType.FLOAT,
    #         strategy=QuantizationStrategy.TENSOR,
    #         symmetric=False,
    #     ),
    # ),
    "linear": QuantizationScheme(targets=["Linear"], **FP8),
}

recipe = [
    SpinQuantModifier(rotations=["R1"], transform_type="random-hadamard"),
    #QuantizationModifier(config_groups=config_groups),
    #QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]

# Apply algorithms.
oneshot(model=model, dataset=ds, recipe=recipe, pipeline="basic")

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
# SAVE_DIR = MODEL_ID.split("/")[1] + "-spinquant-R1R2R4-W4A16"
# model.save_pretrained(SAVE_DIR, save_compressed=True)
# tokenizer.save_pretrained(SAVE_DIR)
