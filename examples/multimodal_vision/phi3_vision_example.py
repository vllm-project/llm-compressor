import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# Load model.
model_id = "microsoft/Phi-3-vision-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    _attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
processor.chat_template = processor.tokenizer.chat_template

# Oneshot arguments
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test[:512]"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


# Apply chat template
def preprocess(example):
    messages = [{"role": "user", "content": "<|image_1|>\nWhat does the image show?"}]
    return {
        "text": processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
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
    sequential_targets=["Phi3DecoderLayer"],
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
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
input_ids = processor(text="Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(processor.decode(output[0]))
print("==========================================")

# Save to disk compressed.
SAVE_DIR = model_id.split("/")[1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
