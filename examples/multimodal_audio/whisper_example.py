import torch
from datasets import load_dataset
from transformers import WhisperProcessor

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.utils.data_collator import whisper_data_collator
from llmcompressor.transformers.tracing import TraceableWhisperForConditionalGeneration

# Select model and load it.
MODEL_ID = "openai/whisper-tiny"

model = TraceableWhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
model.config.forced_decoder_ids = None
processor = WhisperProcessor.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "hf-internal-testing/librispeech_asr_dummy"
DATASET_SPLIT = f"validation[:1]"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 1 # 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, "clean", split=DATASET_SPLIT)


def preprocess(example):
    return {
        "array": example["audio"]["array"],
        "sampling_rate": example["audio"]["sampling_rate"],
    }


ds = ds.map(preprocess, remove_columns=ds.column_names)

r"""
Returns:

Example:
    ```python
    >>> import torch
    >>> from transformers import AutoFeatureExtractor, WhisperModel
    >>> from datasets import load_dataset

    >>> model = WhisperModel.from_pretrained("openai/whisper-base")
    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    >>> input_features = inputs.input_features
    >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
    >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
    >>> list(last_hidden_state.shape)
    [1, 2, 512]
    ```
"""


# Tokenize inputs.
def tokenize(sample):
    batch_size = 1
    input_features = processor(
        sample["array"],
        sampling_rate=sample["sampling_rate"],
        return_tensors="pt",
    ).input_features

    decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long) * model.config.decoder_start_token_id

    return {
        "input_features": input_features,
        "decoder_input_ids": decoder_input_ids
    }


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with GPTQ with a group size 128
#breakpoint()
#sample_input = next(iter(ds))
#output = model(**sample_input)


recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=whisper_data_collator,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
sample_input = whisper_data_collator([next(iter(ds))]).to(model.device)
sample_input = {k: v.to("cuda:0") for k, v in sample_input.items()}
output = model.generate(**sample_input)
breakpoint()
print(processor.batch_decode(output, skip_special_tokens=True))
#[' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.']
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)