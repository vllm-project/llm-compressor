import torch
from datasets import load_dataset
from transformers import AutoProcessor, MllamaForConditionalGeneration, LlavaForConditionalGeneration, AutoModelForCausalLM, Qwen2VLForConditionalGeneration

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
import os

# Load model.
model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

print("Loading dataset")
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test[:512]"

NUM_CALIBRATION_SAMPLES = 1
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

print("Preprocessing samples")
def preprocess(example):
    """
    Preprocesses a single example from the dataset.
    """
    # Example messages structure
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": "What does the image show?"}
            ]
        }
    ]
    return {
        "text": [processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )],
    }

ds = ds.map(preprocess, remove_columns=["caption", "sentids", "img_id", "filename"])


# Tokenize inputs.
def tokenize(sample):
    image = sample.pop("image")
    return processor(
        **sample,
        images=[image],
        add_special_tokens=False, 
        return_tensors="pt"
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

def collate_fn(batch):
    assert len(batch) == 1
    return {
        "input_ids": torch.LongTensor(batch[0]["input_ids"]),
        "attention_mask": torch.tensor(batch[0]["attention_mask"]),
        "pixel_values": torch.tensor(batch[0]["pixel_values"]),  # torch.Size([14308, 1176])
        "image_grid_thw": torch.tensor(batch[0]["image_grid_thw"]),
    }


from llmcompressor.pytorch.utils import tensors_to_device
from llmcompressor.transformers.finetune.data.data_helpers import format_calibration_data
one_sample = next(iter(format_calibration_data(ds, collate_fn=collate_fn)))
batch = tensors_to_device(one_sample, "cuda:0")
model(**batch)

print("Setting up quantization params")
# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
#ignore=["re:.*lm_head", "re:model.vision_embed_tokens.*"]
#ignore=["re:.*lm_head", "re:multi_modal_projector.*", "re:vision_model.*", "re:language_model.*cross_attn.*"],
ignore=["re:.*lm_head", "re:multi_modal_projector.*", "re:vision_model.*"]

recipe = [
    # SmoothQuantModifier(smoothing_strength=0.8, ignore=ignore),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=ignore, update_size=NUM_CALIBRATION_SAMPLES),
]

save_name = model_id.split("/")[1] + "-W8A8"
save_path = os.path.join("./my_test/", save_name)
print("Starting quantization")
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    output_dir=save_path,
)

#processor.save_pretrained(save_path)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
input_ids = processor(text="Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(processor.decode(output[0]))
print("==========================================")
