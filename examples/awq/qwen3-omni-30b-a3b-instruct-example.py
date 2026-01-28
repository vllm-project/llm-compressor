import base64
from io import BytesIO
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modeling.patch.qwen3_omni_patch import fast_pos_embed_interpolate
from llmcompressor.transformers.compression.compressed_tensors_utils import modify_save_pretrained

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
OUTPUT_DIR = MODEL_ID.split("/")[-1] + "-AWQ-W4A16"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 4096

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=None,
    trust_remote_code=True,
)

model.thinker.visual.fast_pos_embed_interpolate = fast_pos_embed_interpolate.__get__(
    model.thinker.visual
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = f"test[:{NUM_CALIBRATION_SAMPLES}]"

ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)

def preprocess_and_tokenize(example):
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue())
    base64_image = f"data:image;base64,{encoded_image.decode('utf-8')}"
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": base64_image},
            {"type": "text", "text": "What does the image show?"}
        ]
    }]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[example["image"]],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )
    return inputs

ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)

def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

recipe = AWQModifier(
    ignore=[
        "re:.*visual.*",
        "re:.*code2wav.*",
        "re:.*audio_tower.*",
        "re:^talker\..*",
        "re:.*embed_tokens",
        "re:.*mlp\.gate$",
        "re:.*shared_expert_gate$",
        "re:.*input_layernorm$",
        "re:.*post_attention_layernorm$",
        "re:.*norm$",
        "re:.*lm_head$"
    ],
    duo_scaling=False,
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "input_activations": None,
            "output_activations": None,
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
                "observer": "mse",
            }
        }
    }
)

oneshot(
    model=model.thinker.model,
    processor=processor,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
    pipeline="sequential", 
)

modify_save_pretrained(model)
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
processor.save_pretrained(OUTPUT_DIR)
