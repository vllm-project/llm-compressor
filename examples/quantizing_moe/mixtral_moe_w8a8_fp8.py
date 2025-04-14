from typing import List

from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map

MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
NUM_GPUS = 2

# Adjust based off number of desired GPUs
device_map = calculate_offload_device_map(
    MODEL_ID, reserve_for_hessians=True, num_gpus=NUM_GPUS, torch_dtype="auto"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map=device_map, torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# Dataset config parameters
DATASET_ID = "open_platypus"
DATASET_SPLIT = "train"
MAX_SEQ_LENGTH = 2048
NUM_CALIBRATION_SAMPLES = 512

# Save location of quantized model
SAVE_DIR = f"{MODEL_ID.split('/')[-1]}-FP8"
SAVE_COMPRESSED = True

layers_to_ignore: List[str] = [
    "lm_head",
    "re:.*block_sparse_moe.gate",  # does not quantize well
]

recipe = QuantizationModifier(scheme="FP8", targets="Linear", ignore=layers_to_ignore)


oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=DATASET_ID,
    splits={"calibration": f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"},
    recipe=recipe,
    max_seq_length=MAX_SEQ_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    save_compressed=SAVE_COMPRESSED,
    output_dir=SAVE_DIR,
)

# Confirm generations of the quantized model look sane.
# Generation is broken for deepseek models when using the latest transformers package
if Version(__version__) < Version("4.48"):
    print("========== SAMPLE GENERATION ==============")
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=20)
    print(tokenizer.decode(output[0]))
    print("==========================================")
else:
    print(
        "WARNING: cannot perform sample generation of "
        "deepseek models with transformers >= 4.48"
    )
