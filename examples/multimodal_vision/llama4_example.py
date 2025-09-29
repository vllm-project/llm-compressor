import torch
from datasets import load_dataset
from transformers import Llama4ForConditionalGeneration, Llama4Processor

from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modifiers.quantization import GPTQModifier

# Select model and load it.
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
model = Llama4ForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto")
processor = Llama4Processor.from_pretrained(model_id)
# We update `Llama4TextMoe` modules with custom `SequentialLlama4TextMoe`.
# This change allows compatibility with vllm.
# To apply your own custom module for experimentation, consider updating
# `SequentialLlama4TextMoe` under llmcompressor/modeling/llama4.py
model = replace_modules_for_calibration(model)

DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 8192

ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess_function(example):
    messgages = []
    for message in example["messages"]:
        messgages.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
        )

    return processor.apply_chat_template(
        messgages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in batch[0].items()
    }


# Configure the quantization algorithm to run.
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "re:.*lm_head",
        "re:.*self_attn",
        "re:.*router",
        "re:.*vision_model.*",
        "re:.*multi_modal_projector.*",
        "Llama4TextAttention",
    ],
)

# Apply algorithms.
# due to the large size of Llama4, we specify sequential targets such that
# only one MLP is loaded into GPU memory at a time
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
    sequential_targets=["Llama4TextMLP"],
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
