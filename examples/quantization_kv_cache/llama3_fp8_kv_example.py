from datasets import load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.transformers import oneshot

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 10
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


def process_and_tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per-tensor scales
#   * quantize the activations to fp8 with per-tensor scales
#   * quantize the kv cache to fp8 with per-tensor scales
recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            config_groups:
                fp8_attention:
                    output_activations:
                        num_bits: 8
                        type: float
                        strategy: channel
                        dynamic: false
                        symmetric: true
                    targets: ['re:.*q_proj', 're:.*k_proj', 're:.*v_proj']
"""
recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            config_groups:
                fp8_attention_q_proj:
                    output_activations:
                        num_bits: 8
                        type: float
                        strategy: group
                        group_size: 512
                        dynamic: false
                        symmetric: true
                    targets: ['re:.*q_proj']
                fp8_attention_kv_proj:
                    output_activations:
                        num_bits: 8
                        type: float
                        strategy: group
                        group_size: 128
                        dynamic: false
                        symmetric: true
                    targets: ['re:.*k_proj', 're:.*v_proj']

"""

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

logger.info(
    "Running sample generation. ",
    "Note: Inference with the quantized kv_cache is not supported. ",
    "Please use vLLM for inference with the quantized kv_cache.",
)
# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-AttnQuantOnly-Group"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
