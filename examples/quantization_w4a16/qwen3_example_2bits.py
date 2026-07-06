from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

# Select model and load it.
model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
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
#   * quantize attention q/k/v/o projections to 2 bit with GPTQ
#   * quantize MLP gate/up/down projections to 4 bit with GPTQ
#   * keep lm_head dense
recipe = GPTQModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=[
                "re:.*self_attn\\.(q|k|v|o)_proj$",
            ],
            weights=QuantizationArgs(
                num_bits=2,
                type="int",
                strategy="group",
                symmetric=True,
                dynamic=False,
                group_size=128,
            ),
        ),
        "mlp": QuantizationScheme(
            targets=[
                "re:.*mlp\\.(gate|up|down)_proj$",
            ],
            weights=QuantizationArgs(
                num_bits=4,
                type="int",
                strategy="group",
                symmetric=True,
                dynamic=False,
                group_size=128,
            ),
        ),
    },
    ignore=["lm_head"],
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Sample generation. Expect some quality loss versus dense / full W4A16, but
# this mixed recipe is materially more stable than full-model 2-bit quantization.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
print("[note] This example quantizes attention to 2-bit and MLP to 4-bit.")
dispatch_model(model)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Introduce yourself in one sentence."}],
    tokenize=False,
    add_generation_prompt=True,
)
sample = tokenizer(prompt, return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-AttnW2A16-MlpW4A16-G128-GPTQ"
print(f"save to {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
