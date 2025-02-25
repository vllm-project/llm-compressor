import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map

# select a Mixture of Experts model for quantization
MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B-Chat"

# adjust based off number of desired GPUs
# if not enough memory is available, some layers will automatically be offloaded to cpu
device_map = calculate_offload_device_map(
    MODEL_ID,
    reserve_for_hessians=True,
    num_gpus=2,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map=device_map, torch_dtype=torch.bfloat16, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


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

# define a llmcompressor recipe for W416 quantization with a group size of 128
# since the MoE gate layers are sensitive to quantization, we add them to the ignore
# list so they remain at full precision
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"],
)

SAVE_DIR = MODEL_ID.split("/")[1] + "-quantized.w4a16"


oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    save_compressed=True,
    trust_remote_code_model=True,
    output_dir=SAVE_DIR,
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")
