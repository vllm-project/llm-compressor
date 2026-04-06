# Gemma 4 requires transformers >= 5.5.0 (model_type: gemma4).
# If your llmcompressor pins an older version, install with:
#   pip install --no-deps llmcompressor
#   pip install git+https://github.com/huggingface/transformers.git

# Checkpoint available at https://huggingface.co/RedHatAI/gemma-4-31B-it-NVFP4

import requests
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, Gemma4ForConditionalGeneration, ProcessorMixin

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Load model.
model_id = "google/gemma-4-31B-it"
model = Gemma4ForConditionalGeneration.from_pretrained(model_id, dtype="auto")
processor = AutoProcessor.from_pretrained(model_id)

# Oneshot arguments
BATCH_SIZE = 1
NUM_CALIBRATION_SAMPLES = 32
MAX_SEQUENCE_LENGTH = 2048

DATASET_ID = "mit-han-lab/pile-val-backup"
DATASET_SPLIT = "validation"

# Recipe
recipe = [
    QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=["re:.*vision.*", "re:.*audio.*", "lm_head", "re:.*embed.*"],
    ),
]


def get_calib_dataset(processor: ProcessorMixin):
    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES*10}]",
    )

    def preprocess(example):
        return {
            "input_ids": processor.tokenizer.encode(example["text"].strip())[
                :MAX_SEQUENCE_LENGTH
            ]
        }

    ds = (
        ds.shuffle(seed=42)
        .map(preprocess, remove_columns=ds.column_names)
        .filter(lambda example: len(example["input_ids"]) >= MAX_SEQUENCE_LENGTH)
        .select(range(NUM_CALIBRATION_SAMPLES))
    )

    return ds


# Perform oneshot
oneshot(
    model=model,
    processor=processor,
    dataset=get_calib_dataset(processor),
    recipe=recipe,
    batch_size=BATCH_SIZE,
    shuffle_calibration_samples=False,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
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

# Note: compile is disabled: https://github.com/huggingface/transformers/issues/38333
inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100, disable_compile=True)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
