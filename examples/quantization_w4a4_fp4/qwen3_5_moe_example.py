import os
import shutil

from datasets import concatenate_datasets, load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

# Available Qwen3.5 MoE models (pick one):
#   "Qwen/Qwen3.5-35B-A3B"
#   "Qwen/Qwen3.5-122B-A10B"
#   "Qwen/Qwen3.5-397B-A17B"
MODEL_ID = "Qwen/Qwen3.5-35B-A3B"

# Load model.
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select number of samples. 512 is recommended for production quality;
# reduce to 256 or lower for faster iteration during development.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 4096

# Load datasets and preprocess.
# Use half from each source for a diverse calibration set.
samples_per_dataset = NUM_CALIBRATION_SAMPLES // 2

ds_ultrachat = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split=f"train_sft[:{samples_per_dataset}]",
)
ds_nemotron = load_dataset(
    "nvidia/Nemotron-Post-Training-Dataset-v2",
    split=f"chat[:{samples_per_dataset}]",
)

# Both datasets share a "messages" column with the same chat format.
# Keep only that column so we can concatenate them.
ds_ultrachat = ds_ultrachat.select_columns(["messages"])
ds_nemotron = ds_nemotron.select_columns(["messages"])
ds = concatenate_datasets([ds_ultrachat, ds_nemotron])
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
# In this case, we:
#   * quantize the weights to fp4 with per group 16 via ptq
#   * calibrate a global_scale for activations, which will be used to
#       quantize activations to fp4 on the fly
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        "re:.*mlp.gate$",
        "re:.*mlp.shared_expert_gate$",
        "re:.*linear_attn.*",
        "re:model\\.visual\\..*",
    ],
)

# Apply quantization.
# MoE calibration is now handled automatically by the pipeline.
# We set `moe_calibrate_all_experts` to True to ensure all experts receive
# calibration data. This temporarily updates the model definition to use
# `CalibrationQwen3_5MoeSparseMoeBlock` (from `llmcompressor.modeling.qwen3_5_moe`)
# which replaces the original `Qwen3_5MoeSparseMoeBlock` class.
# This unfuses the 3D expert parameters into individual nn.Linear modules
# so they can be targeted by quantization.
# Feel free to update the definition under
# llm-compressor/src/llmcompressor/modeling/qwen3_5_moe.py to play around with
# this behavior and evaluate its impact on quantization performance.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=True,
)


print("\n\n")
print("========== SAMPLE GENERATION ==============")
try:
    dispatch_for_generation(model)
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
        model.device
    )
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
except Exception as e:
    print(f"Generation failed (non-fatal): {e}")
print("==========================================\n\n")


# Save to disk in compressed-tensors format.
# MTP (multi-token prediction) weights are automatically preserved from
# the source checkpoint during save, enabling speculative decoding with vLLM.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

# Hot-fix: copy processor configs that save_pretrained doesn't bring over
cache_dir = snapshot_download(MODEL_ID, allow_patterns=["*.json"])
for filename in [
    "preprocessor_config.json",
    "video_preprocessor_config.json",
]:
    src = os.path.join(cache_dir, filename)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(SAVE_DIR, filename))
