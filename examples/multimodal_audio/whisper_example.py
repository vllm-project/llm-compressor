from datasets import load_dataset
from transformers import WhisperProcessor

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.tracing import TraceableWhisperForConditionalGeneration
from llmcompressor.transformers.utils.data_collator import whisper_data_collator

# Select model and load it.
MODEL_ID = "openai/whisper-base"

model = TraceableWhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
model.config.forced_decoder_ids = None
processor = WhisperProcessor.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "MLCommons/peoples_speech"
DATASET_SUBSET = "test"
DATASET_SPLIT = "test"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(
    DATASET_ID,
    DATASET_SUBSET,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
    trust_remote_code=True,
)


def preprocess(example):
    return {
        "array": example["audio"]["array"],
        "sampling_rate": example["audio"]["sampling_rate"],
    }


ds = ds.map(preprocess, remove_columns=ds.column_names)


# Process inputs.
def process(sample):
    input_features = processor(
        sample["array"],
        sampling_rate=sample["sampling_rate"],
        return_tensors="pt",
    ).input_features

    # decoder_input_ids define the task context
    generation_config, _kwargs = model._prepare_generation_config(None)
    input_stride = (
        model.model.encoder.conv1.stride[0] * model.model.encoder.conv2.stride[0]
    )
    num_segment_frames = input_stride * model.config.max_source_positions
    decoder_input_ids = model._retrieve_init_tokens(
        input_features,
        batch_size=1,
        generation_config=generation_config,
        config=model.config,
        num_segment_frames=num_segment_frames,
        kwargs={},
    )

    return {"input_features": input_features, "decoder_input_ids": decoder_input_ids}


ds = ds.map(process, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with GPTQ with a group size 128
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
sample_input = whisper_data_collator([next(iter(ds))])
sample_input = {k: v.to(model.device) for k, v in sample_input.items()}
output = model.generate(**sample_input, language="en")
print(processor.batch_decode(output, skip_special_tokens=True))
print("==========================================\n\n")
# The track appears on the compilation album "Kraftworks"

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
