import base64
from io import BytesIO

import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset

# Note: this is an optional utility for processing vision inputs for qwen.
# This can be installed via the "qwen" extra
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

# Load model.
model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# Oneshot arguments
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test[:512]"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)


# Apply chat template and tokenize inputs.
def preprocess_and_tokenize(example):
    # preprocess
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": base64_qwen},
                {"type": "text", "text": "What does the image show?"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # tokenize
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )


ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    token_fields = ("input_ids", "attention_mask", "mm_token_type_ids")
    concat_fields = ("pixel_values", "image_grid_thw")

    result = {}
    for key in token_fields:
        if key not in batch[0]:
            continue
        tensors = [torch.tensor(sample[key]).squeeze(0) for sample in batch]
        max_len = max(t.shape[0] for t in tensors)
        padded = torch.zeros(len(tensors), max_len, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            padded[i, : t.shape[0]] = t
        result[key] = padded

    for key in concat_fields:
        if key not in batch[0]:
            continue
        result[key] = torch.cat([torch.tensor(sample[key]) for sample in batch], dim=0)

    return result


# Recipe
recipe = [
    GPTQModifier(
        scheme="W4A16",
        ignore=["re:.*lm_head", "re:.*visual.*"],
    ),
]
# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
    batch_size=4,
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "http://images.cocodataset.org/train2017/000000231895.jpg",
            },
            {"type": "text", "text": "Please describe the animal in this image\n"},
        ],
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=False,
    max_length=MAX_SEQUENCE_LENGTH,
    truncation=True,
    return_tensors="pt",
).to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================")


# Save to disk compressed.
# NOTE: for now, must use `save_original_format` to avoid errors with repacking qparams
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16"
model.save_pretrained(SAVE_DIR, save_compressed=True, save_original_format=False)
processor.save_pretrained(SAVE_DIR)
