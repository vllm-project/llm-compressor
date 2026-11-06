from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context

# Select model and load it. Qwen3.5-9B is a multimodal model with a vision
# tower and an MTP head. The lm_head and vision tower are kept in bfloat16.
# MTP calibration is not yet supported, so the MTP layers are not loaded
# (load_mtp=False) and are copied unchanged into the saved checkpoint.
MODEL_ID = "Qwen/Qwen3.5-9B"

with load_context(Qwen3_5ForConditionalGeneration, load_mtp=False):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 1024
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

# Configure the quantization algorithm to run.
#   * quantize the weights and activations to NVFP4 with GPTQ
#   * use the nvfp4_expanded_imatrix weight observer, which weights
#     quantization error by per-input-channel activation importance (E[x^2])
#     collected during calibration and searches NVFP4 range expansions from
#     1.8x down to ~0.8x of the per-group min/max range
recipe = GPTQModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head", "re:.*visual.*"],
    weight_observer="nvfp4_expanded_imatrix",
)

# Apply quantization.
oneshot(
    model=model,
    processor=processor,
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
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-GPTQ-expanded-imatrix"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
