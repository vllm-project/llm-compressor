import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
MODEL_ID = "Qwen/Qwen2-Audio-7B"
model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "MLCommons/peoples_speech"
DATASET_SUBSET = "test"
DATASET_SPLIT = "test"

NUM_CALIBRATION_SAMPLES = 4#512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(
    DATASET_ID,
    DATASET_SUBSET,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
)

def preprocess(example):
    # Peoples Speech: example["audio"] = {"array": ..., "sampling_rate": ...}
    # example["text"] is transcript
    return {
        "array": example["audio"]["array"],
        "sampling_rate": example["audio"]["sampling_rate"],
        "text": example["text"].strip(),
    }

ds = ds.map(preprocess, remove_columns=ds.column_names)

# Process inputs.
PROMPT_PREFIX = "<|audio_bos|><|AUDIO|><|audio_eos|>Transcribe the audio in English:"

def process(sample):
    text = f"{PROMPT_PREFIX} {sample['text']}"

    # 1) Audio -> padded mel features (exactly 3000 frames)
    audio_feats = processor.feature_extractor(
        sample["array"],
        sampling_rate=sample["sampling_rate"],
        padding="max_length",
        max_length=processor.feature_extractor.n_samples,
        return_tensors="pt",
    )

    # 2) Text -> token ids (your chosen MAX_SEQUENCE_LENGTH)
    text_toks = processor.tokenizer(
        text,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    )

    # Merge to what Qwen2AudioForConditionalGeneration expects
    inputs = {
        "input_features": audio_feats["input_features"][0],   # strip batch dim
        "input_ids": text_toks["input_ids"][0],
        "attention_mask": text_toks["attention_mask"][0],
    }
    # Some versions also provide/expect a feature_attention_mask; include if present
    if "attention_mask" in audio_feats and audio_feats["attention_mask"] is not None:
        inputs["feature_attention_mask"] = audio_feats["attention_mask"][0]

    return inputs

ds = ds.map(process, remove_columns=ds.column_names)

# Recipe
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],  # safe default for generation heads
)

# Apply algorithms.
# oneshot(
#     model=model,
#     dataset=ds,
#     recipe=recipe,
#     max_seq_length=MAX_SEQUENCE_LENGTH,
#     num_calibration_samples=NUM_CALIBRATION_SAMPLES,
# )

# Confirm generations of the quantized model look sane.
print("\n========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
sample = next(iter(ds))
sample = {key: torch.tensor([value]).to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
text = processor.batch_decode(output, skip_special_tokens=True)[0]
print(text)
print("==========================================\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-GPTQ"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
print(f"Saved to: {SAVE_DIR}")
