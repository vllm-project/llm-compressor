from transformers import AutoModelForImageTextToText, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

# Load model.
model_id = "OpenGVLab/InternVL3-8B-hf"
model = AutoModelForImageTextToText.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# Oneshot arguments
DATASET_ID = "flickr30k"
DATASET_SPLIT = "test"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Recipe
recipe = GPTQModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["re:.*lm_head", "re:.*vision_tower.*", "re:.*multi_modal_projector.*"],
)

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=DATASET_ID,
    splits=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
