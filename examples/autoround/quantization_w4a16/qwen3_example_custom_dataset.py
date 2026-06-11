from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier, fix_batch_if_needed
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_ID = "Qwen/Qwen3-8B"
# MODEL_ID = "Qwen/Qwen3-0.6B"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048
ITERS = 200
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
).shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
        return_attention_mask=True,
    )


dataset = dataset.map(preprocess)
dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
dataset = dataset.map(fix_batch_if_needed)

recipe = AutoRoundModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
    iters=ITERS,
)

oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("\n\n========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

save_dir = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-G128-AutoRound-Ultrachat200k"
model.save_pretrained(save_dir, save_compressed=True)
tokenizer.save_pretrained(save_dir)
