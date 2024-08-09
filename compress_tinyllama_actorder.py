import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
def get_current_time():
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")
    return str(formatted_time)
# Select model and load it.
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = SparseAutoModelForCausalLM.from_pretrained(
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
NUM_CALIBRATION_SAMPLES = 1
MAX_SEQUENCE_LENGTH = 512
# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))
def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }
ds = ds.map(preprocess)
# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )
ds = ds.map(tokenize, remove_columns=ds.column_names)
recipe = """
    quant_stage:
        quant_modifiers:
            GPTQModifier:
                sequential_update: false
                ignore: ["lm_head"]
                config_groups:
                    group_0:
                        weights:
                            num_bits: 4
                            type: "int"
                            symmetric: true
                            strategy: "group"
                            group_size: 128
                            actorder: True
                        targets: ["Linear"]
"""
# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
breakpoint()

# save model
SAVE_DIR = "actorder" + get_current_time()
print(SAVE_DIR)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR, save_compressed=True)
breakpoint()

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
# model.get_parameter("model.layers.9.mlp.up_proj.weight_g_idx")
# model.name_or_path
# breakpoint()
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")
# # breakpoint()
# # # Save to disk compressed.
# # SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A16-G128-ACTORDER"
# # SAVE_DIR = "actorder_v5"