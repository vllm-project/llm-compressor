from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import preset_name_to_scheme
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.imatrix import IMatrixGatherer

# Select model and load it.
model_id = "meta-llama/Meta-Llama-3.1-8B"

model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
DATASET_ID = "open_platypus"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Configure the quantization algorithm to run.
#   * trigger a calibration pass with IMatrixGatherer so the observer can collect E[x²]
#   * quantize the weights to 4 bit with group size 128
#   * use imatrix_mse observer to weight quantization error by channel importance
scheme = preset_name_to_scheme("W4A16", ["Linear"])
scheme.weights.observer = "imatrix_mse"

recipe = [
    IMatrixGatherer(ignore=["lm_head"]),
    QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=DATASET_ID,
    splits={"calibration": "train[:5%]"},
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128-imatrix"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
