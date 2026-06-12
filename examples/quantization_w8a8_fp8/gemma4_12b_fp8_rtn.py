# Quantize gemma-4-12B-it to FP8 using RTN (round-to-nearest)
#
# Requires transformers >= 5.0 for correct weight name mapping in config.json.
#
# Evaluate with e.g.:
#   lm_eval --model vllm \
#     --model_args "pretrained=gemma-4-12B-it-FP8-RTN,\
#       dtype=auto,max_model_len=4096,add_bos_token=True,\
#       gpu_memory_utilization=0.85" \
#     --tasks gsm8k_platinum --num_fewshot 5 \
#     --apply_chat_template --batch_size auto

import torch
import transformers
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset

if int(transformers.__version__.split(".")[0]) < 5:
    raise RuntimeError(
        f"transformers >= 5.0 required, found {transformers.__version__}"
    )

from transformers import AutoModelForImageTextToText, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "google/gemma-4-12B-it"
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess_function(example):
    messages = []
    for message in example["messages"]:
        messages.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
        )

    return processor.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    if len(batch) != 1:
        raise ValueError(f"Expected batch size 1, got {len(batch)}")
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in batch[0].items()
    }


recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "lm_head",
        "re:.*embed_vision.*",
        "re:.*embed_audio.*",
        "re:.*vision_embedder.*",
    ],
)

oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = torch.tensor(
    [
        [
            2,
            105,
            2364,
            107,
            818,
            3282,
            506,
            7217,
            563,
            3730,
            563,
            1547,
            106,
            107,
            105,
            4368,
            107,
        ]
    ]
).to(model.device)
output = model.generate(
    input_ids,
    max_new_tokens=100,
)
print(processor.tokenizer.decode(output[0]))
print("==========================================\n\n")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-RTN"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
