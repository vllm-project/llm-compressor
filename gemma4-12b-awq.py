import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

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
    assert len(batch) == 1
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in batch[0].items()
    }


recipe = [
    AWQModifier(),
    QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            "lm_head",
            "re:.*embed_vision.*",
            "re:.*embed_audio.*",
            "re:.*vision_embedder.*",
        ],
    ),
]

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
    [[
        2, 105, 2364, 107, 818, 3282, 506, 7217, 563, 3730, 563,
        1547, 106, 107, 105, 4368, 107
    ]]
).to(model.device)
output = model.generate(
    input_ids,
    max_new_tokens=1000,
)
print(processor.tokenizer.decode(output[0]))
print("==========================================\n\n")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-AWQ-bad"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

# Patch config: transformers renames checkpoint keys on load (vision_embedder ->
# embed_vision), but save_pretrained reverts them. The ignore list in config.json
# uses HF names (embed_vision) while safetensors keys use checkpoint names
# (vision_embedder), so vllm can't match them. Add the checkpoint name explicitly.
import json as _json
_cfg_path = SAVE_DIR + "/config.json"
with open(_cfg_path) as _f:
    _cfg = _json.load(_f)
_qcfg = _cfg.get("quantization_config")
if _qcfg:
    _ign = _qcfg.setdefault("ignore", [])
    if "model.vision_embedder.patch_dense" not in _ign:
        _ign.append("model.vision_embedder.patch_dense")
        with open(_cfg_path, "w") as _f:
            _json.dump(_cfg, _f, indent=2)
        print("Patched config.json: added vision_embedder.patch_dense to ignore list")

if False:
    """
        lm_eval --model vllm \
            --model_args "pretrained=gemma-4-12B-it-NVFP4-AWQ,dtype=auto,max_model_len=4096,add_bos_token=True,gpu_memory_utilization=0.85" \
            --tasks "mmlu" \
            --num_fewshot 5 \
            --apply_chat_template \
            --batch_size auto \
            --output_path "gemma-4-12B-it-NVFP4-AWQ_eval.json"
    """