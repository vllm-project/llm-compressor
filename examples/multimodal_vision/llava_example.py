from transformers import AutoProcessor

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.tracing import TracableLlavaForConditionalGeneration
from llmcompressor.transformers.utils.data_collator import llava_data_collator

# Load model.
model_id = "llava-hf/llava-1.5-7b-hf"
model = TracableLlavaForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto"
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Oneshot arguments
DATASET_ID = "flickr30k"
DATASET_SPLIT = {"calibration": "test[:512]"}
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["re:.*lm_head", "re:vision_tower.*", "re:multi_modal_projector.*"],
        sequential_targets=["LlamaDecoderLayer"],
    ),
]

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=DATASET_ID,
    splits=DATASET_SPLIT,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=llava_data_collator,
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
