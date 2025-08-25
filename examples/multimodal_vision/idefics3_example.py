import requests
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

# Load model.
model_id = "HuggingFaceM4/Idefics3-8B-Llama3"  # or "HuggingFaceTB/SmolVLM-Instruct"
model = Idefics3ForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Oneshot arguments
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test[:512]"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 4096  # Seems to be required here


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["re:.*lm_head", "re:model.vision_model.*", "re:model.connector.*"],
    ),
]

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


# Apply chat template
def preprocess(example):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What does this image show?"},
                {"type": "image"},
            ],
        },
        {
            "role": "assistant",
            "content": " ".join(example["caption"]),
        },
    ]
    return {
        "text": processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
        ),
        "images": example["image"],
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return processor(
        text=sample["text"],
        images=sample["images"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )


# avoid errors with writer_batch_size
ds = ds.map(tokenize, writer_batch_size=1, remove_columns=ds.column_names)

# Perform oneshot
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
    sequential_targets=["LlamaDecoderLayer"],
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please describe the animal in this image\n"},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
image_url = "http://images.cocodataset.org/train2017/000000231895.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw)

inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
