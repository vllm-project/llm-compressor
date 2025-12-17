import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

# Enable CUDA synchronous execution for better error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 64
MAX_SEQUENCE_LENGTH = 200

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


# Configure the quantization algorithm to run.
recipe = [
    AWQModifier(
        ignore=["lm_head"], scheme="NVFP4A16", targets=["Linear"], duo_scaling="both"
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("\n\n")
print("=" * 80)
print("VALIDATION: Checking model weights for inf/nan values")
print("=" * 80)

# Validate model weights
has_invalid_values = False
for name, param in model.named_parameters():
    # Skip meta tensors
    if param.device == torch.device("meta"):
        continue
    # Convert to float32 for dtypes that don't support isfinite (like Float8_e4m3fn)
    param_to_check = param.float() if param.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else param
    if not param_to_check.isfinite().all():
        print(f"❌ INVALID: Found inf/nan in parameter: {name}")
        print(f"   Dtype: {param.dtype}")
        print(f"   Shape: {param.shape}")
        print(f"   Min: {param_to_check.min()}, Max: {param_to_check.max()}")
        print(f"   Inf count: {torch.isinf(param_to_check).sum()}")
        print(f"   NaN count: {torch.isnan(param_to_check).sum()}")
        has_invalid_values = True

# Check buffers (like weight_scale, weight_zero_point)
for name, buffer in model.named_buffers():
    if buffer is not None:
        # Skip meta tensors
        if buffer.device == torch.device("meta"):
            continue
        # Convert to float32 for dtypes that don't support isfinite (like Float8_e4m3fn)
        buffer_to_check = buffer.float() if buffer.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] else buffer
        if not buffer_to_check.isfinite().all():
            print(f"❌ INVALID: Found inf/nan in buffer: {name}")
            print(f"   Dtype: {buffer.dtype}")
            print(f"   Shape: {buffer.shape}")
            print(f"   Min: {buffer_to_check.min()}, Max: {buffer_to_check.max()}")
            print(f"   Inf count: {torch.isinf(buffer_to_check).sum()}")
            print(f"   NaN count: {torch.isnan(buffer_to_check).sum()}")
            has_invalid_values = True

if not has_invalid_values:
    print("✅ All weights and buffers are valid (no inf/nan values)")
else:
    print("\n⚠️  WARNING: Invalid values detected! Generation may fail.")

print("=" * 80)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
try:
    dispatch_for_generation(model)
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
        model.device
    )

    # Test forward pass first
    print("Testing forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

        if not logits.isfinite().all():
            print(f"❌ ERROR: Model produces invalid logits!")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Inf count: {torch.isinf(logits).sum()}")
            print(f"   NaN count: {torch.isnan(logits).sum()}")
            print(f"   Min: {logits.min()}, Max: {logits.max()}")
        else:
            print(f"✅ Forward pass produces valid logits")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Min: {logits.min():.4f}, Max: {logits.max():.4f}")

    print("\nAttempting generation...")
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("✅ Generation successful!")
except Exception as e:
    print(f"❌ Generation failed with error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("==========================================\n\n")

# Save to disk compressed.
base_name = MODEL_ID.rstrip("/").split("/")[-1] + "-awq"
hf_cache = os.environ.get("HF_HUB_CACHE")
if hf_cache and os.path.isdir(hf_cache):
    SAVE_DIR = os.path.join(hf_cache, base_name)
else:
    SAVE_DIR = base_name

model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
