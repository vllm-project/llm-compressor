import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

import math
import numpy as np
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.utils import dispatch_for_generation


# IMPORTANT: Before running this script, you must manually modify the
# `modeling_internvl_chat.py` file in your local copy of the `OpenGVLab/InternVL3-8B`
# model directory.
# Replace the original `forward` method of the `InternVLChatModel` class with the
# version provided in `examples/multimodal_vision/internvl3_README.md`.
# This step is necessary to disable vision processing during calibration.

# Load model.
model_id = "OpenGVLab/InternVL3-8B"
model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

# Oneshot arguments
DATASET_ID = "flickr30k"
DATASET_SPLIT = {"calibration": "test[:512]"}
NUM_CALIBRATION_SAMPLES = 4  # NOTE: Using a small number for a quick example. For best results, increase to 512 or more.
MAX_SEQUENCE_LENGTH = 2048
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)


#### copy from https://hf-mirror.com/OpenGVLab/InternVL3-8B#inference-with-transformers ####
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
#### copy from https://hf-mirror.com/OpenGVLab/InternVL3-8B#inference-with-transformers end ####


def load_image_from_PIL(image_obj, input_size=448, max_num=12):
    image = image_obj.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def preprocess_and_tokenize(example):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": ""
                },
                {
                    "type": "text", 
                    "text":  "What does the image show?"
                },
            ],
        }
    ]
    text = tokenizer.apply_chat_template(messages)

    example["input_ids"] = text
    return example

ds = ds.map(preprocess_and_tokenize)

# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    item = {key: value for key, value in batch[0].items()}
    item["pixel_values"] = load_image_from_PIL(item["image"])
    item["labels"] = torch.LongTensor([item["input_ids"]])
    item["input_ids"] = torch.LongTensor([item["input_ids"]])
    return item

# Recipe
recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["re:.*lm_head",  "re:mlp1.*"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""

# Perform oneshot

oneshot(
    model=model,
    tokenizer=model_id,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
)


# Save to disk compressed.
SAVE_DIR = "./InternVL3-8B-FP8_W8A8-FP8_KV"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)