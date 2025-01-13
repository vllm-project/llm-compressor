from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot

#MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load model.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
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


recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            transforms:
                weights:
                    type: identical
                    targets: ["Linear"]
                    transpose: false
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: channel
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
"""

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
