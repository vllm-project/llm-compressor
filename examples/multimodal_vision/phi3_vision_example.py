# NOTE: this model requires modification in order to work with transformers>4.48
# https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/discussions/69

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

# Load model.
model_id = "microsoft/Phi-3-vision-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    trust_remote_code=True,
    _attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
processor.chat_template = processor.tokenizer.chat_template

# Oneshot arguments
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


# Apply chat template
def preprocess(example):
    messages = [
        {"role": "user", "content": "<|image_1|>\nWhat does this image show?"},
        {"role": "assistant", "content": " ".join(example["caption"])},
    ]
    return {
        "text": processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
        ),
        "images": example["image"],
    }


ds = ds.map(preprocess)


# # Tokenize inputs.
def tokenize(sample):
    return processor(
        text=sample["text"],
        images=sample["images"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )


# long data lengths produced by the phi3_vision processor
# can lead to integer overflows when mapping, avoid with writer_batch_size
ds = ds.map(tokenize, writer_batch_size=1, remove_columns=ds.column_names)


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


# Recipe
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head", "re:model.vision_embed_tokens.*"],
)

# Perform oneshot
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
    sequential_targets=["Phi3DecoderLayer"],
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = processor(text="Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(processor.decode(output[0]))
print("==========================================")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
