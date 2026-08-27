from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier, fix_batch_if_needed
from llmcompressor.modifiers.quantization import QuantizationModifier

model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 1024
ITERS = 0

ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
        return_attention_mask=True,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)
ds = ds.map(fix_batch_if_needed)

recipe = [
    QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=["Qwen3MoeAttention"],
                input_activations=QuantizationArgs(
                    num_bits=8, type="float", strategy="tensor"
                ),
            ),
        },
    ),
    AutoRoundModifier(
        targets="Linear",
        scheme="MXFP8",
        ignore=["lm_head", "re:.*mlp.gate$"],
        iters=ITERS,
    ),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=50)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-FP8Attention-MXFP8-AutoRound"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print("Saved to", SAVE_DIR)
