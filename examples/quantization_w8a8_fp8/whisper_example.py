from datasets import load_dataset
from transformers import AutoProcessor, WhisperForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "openai/whisper-large-v2"

# Load model.
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto")
model.config.forced_decoder_ids = None
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]"
)
sample = ds[0]["audio"]
input_features = processor(
    sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
).input_features
input_features = input_features.to(model.device)
output_ids = model.generate(input_features, language="en", forced_decoder_ids=None)
print(processor.batch_decode(output_ids, skip_special_tokens=False)[0])
# Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
