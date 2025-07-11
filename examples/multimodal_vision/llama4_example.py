from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.modeling import prepare_for_calibration
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = prepare_for_calibration(model)

# Select calibration dataset.
DATASET_ID = "flickr30k"
DATASET_SPLIT = "test"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Configure the quantization algorithm to run.
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "re:.*lm_head",
        "re:.*router",
        "re:vision_model.*",
        "re:multi_modal_projector.*",
    ],
)

# Apply algorithms.
# due to the large size of Llama4, we specify sequential targets such that
# only one MLP is loaded into GPU memory at a time
oneshot(
    model=model,
    dataset=DATASET_ID,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    sequential_targets=["Llama4TextAttention", "Llama4TextMLP"],
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to("cuda") for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
