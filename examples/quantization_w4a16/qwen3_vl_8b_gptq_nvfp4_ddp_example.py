###############################################################################
# This script quantizes Qwen3-VL-8B with GPTQ + NVFP4 using DDP.
# run this with `torchrun --nproc_per_node=2 qwen3_vl_8b_gptq_nvfp4_ddp_example.py`
# or change nproc_per_node to your desired configuration
###############################################################################

import base64
import time
from io import BytesIO

import torch
import torch.distributed as dist
from compressed_tensors.offload import dispatch_model, init_dist, load_offloaded_model
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.quantization import GPTQModifier

model_id = "Qwen/Qwen3-VL-8B-Instruct"

###### DDP MODEL LOAD CHANGE #####
init_dist()
with load_offloaded_model():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype="auto", device_map="auto_offload"
    )
##################################

processor = AutoProcessor.from_pretrained(model_id)

# Oneshot arguments
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

###### DDP DATA LOAD CHANGE #####
ds = load_dataset(
    DATASET_ID, split=get_rank_partition(DATASET_SPLIT, NUM_CALIBRATION_SAMPLES)
)
##################################

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
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


# Recipe: GPTQ + NVFP4
recipe = GPTQModifier(
    targets="Linear",
    scheme="NVFP4A16",
    ignore=["re:.*lm_head", "re:.*visual.*"],
)

torch.cuda.reset_peak_memory_stats()
start_time = time.time()

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
    sequential_targets=["Qwen3VLTextDecoderLayer"],
)

elapsed_time = time.time() - start_time
peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
print("Quantization Complete")
print(f"Time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")
print(f"Peak GPU Memory: {peak_memory_gb:.2f} GB")

# Confirm generations of the quantized model look sane.
print("\n\n")
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
print("==========================================\n\n")

print("Saving...")
# Save to disk compressed.
SAVE_DIR = (
    model_id.rstrip("/").split("/")[-1]
    + "-GPTQ-NVFP4A16-DDP"
    + str(torch.distributed.get_world_size())
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

dist.destroy_process_group()
