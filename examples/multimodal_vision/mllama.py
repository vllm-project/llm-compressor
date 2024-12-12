import os

import torch
from transformers import AutoProcessor

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.pytorch.data_collator import DataCollator
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.tracing import TracableMllamaForConditionalGeneration

# Load model.
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = TracableMllamaForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto"
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Oneshot arguments
DATASET_ID = "flickr30k"
DATASET_SPLIT = "test[:512]"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


# TODO: define real collators in utils
def data_collator(batch):
    assert len(batch) == 1
    return {
        "input_ids": torch.LongTensor(batch[0]["input_ids"]),
        "attention_mask": torch.tensor(batch[0]["attention_mask"]),
        "pixel_values": torch.tensor(batch[0]["pixel_values"]),
        "aspect_ratio_ids": torch.tensor(batch[0]["aspect_ratio_ids"]),
        "aspect_ratio_mask": torch.tensor(batch[0]["aspect_ratio_mask"]),
        "cross_attention_mask": torch.tensor(batch[0]["cross_attention_mask"]),
    }


# Recipe
recipe = [
    # SmoothQuantModifier(smoothing_strength=0.8, ignore=ignore),
    GPTQModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=["re:.*lm_head", "re:multi_modal_projector.*", "re:vision_model.*"],
        dampening_frac=100.0,
    ),
]

# Perform oneshot
save_name = model_id.split("/")[1] + "-W8A8"
save_path = os.path.join("./my_test/", save_name)
print("Starting quantization")
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=DATASET_ID,
    splits=DATASET_SPLIT,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    output_dir=save_path,
    data_collator=data_collator,
    # data_collator=DataCollator(),
)

processor.save_pretrained(save_path)
model.save_pretrained(save_path)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
input_ids = processor(text="Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(processor.decode(output[0]))
print("==========================================")
