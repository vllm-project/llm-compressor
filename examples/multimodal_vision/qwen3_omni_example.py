import requests
import soundfile as sf
from compressed_tensors.offload import dispatch_model
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3OmniMoeForConditionalGeneration,
    default_data_collator,
)

from llmcompressor import oneshot
from llmcompressor.modeling.patch.qwen3_omni_patch import fast_pos_embed_interpolate
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)

# Load model.
model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto"
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Apply patch to fix accelerate offloading, can be removed after #2148
model.thinker.visual.fast_pos_embed_interpolate = fast_pos_embed_interpolate.__get__(
    model.thinker.visual
)

# Oneshot arguments
BATCH_SIZE = 1
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048
DATASET_ID = "flickr30k"
DATASET_SPLIT = {"calibration": f"test[:{NUM_CALIBRATION_SAMPLES}]"}

# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
            "lm_head",
            r"re:.*visual.*",
            r"re:.*code2wav.*",
        ],
    ),
]


def data_collator(features):
    batch = default_data_collator(features)
    batch["image_grid_thw"] = batch["image_grid_thw"].squeeze(0)
    return batch


# Perform oneshot
oneshot(
    model=model.thinker,  # base model does not define forward: pass `thinker` instead
    processor=processor,
    dataset=DATASET_ID,
    splits=DATASET_SPLIT,
    recipe=recipe,
    batch_size=BATCH_SIZE,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
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

inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(model.device)
text_ids, audio = model.generate(**inputs, max_new_tokens=100, disable_compile=True)
text = processor.batch_decode(
    text_ids.sequences[:, inputs["input_ids"].shape[1] :],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(text)
if audio is not None:
    sf.write(
        "sample_output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )
print("==========================================")

# Save to disk compressed.
modify_save_pretrained(model)
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
