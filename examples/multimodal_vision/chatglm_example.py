import requests
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from datasets import load_dataset
from llmcompressor.transformers.tracing import TraceableChatGLMForConditionalGeneration

from llmcompressor.transformers.utils.data_collator import glm_data_collator

MODEL_ID = "THUDM/glm-4v-9b"
model = TraceableChatGLMForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

NUM_CALIBRATION_SAMPLES = 1
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

def preprocess(example):
    url_part = "/".join(example["image"].split("/")[1:])
    url = f"http://images.cocodataset.org/{url_part}"
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert('RGB')

    return processor.apply_chat_template(
        [
            {
                "role": "user",
                "image": image,
                "content": example["conversations"][0]["value"],
            }
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )


ds = ds.map(preprocess, remove_columns=ds.column_names)

# Configure the quantization algorithms
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        sequential_targets=["GLMBlock"],
        ignore=["transformer.output_layer", "re:transformer.vision.*"],
        dampening_frac=1e10,
    ),
]

# # Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=glm_data_collator,
    output_dir="my_glm_save"
)
# from accelerate.accelerator import get_state_dict_offloaded_model

import torch
torch.cuda.memory._record_memory_history()
#state_dict = get_state_dict_offloaded_model(model)
model.save_pretrained("asdf")
torch.cuda.memory._dump_snapshot(f"vanilla.pickle")