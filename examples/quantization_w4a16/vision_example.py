from datasets import load_dataset
from transformers import AutoProcessor, MllamaForConditionalGeneration

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot

# Select model and load it.
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="cuda:0",
    torch_dtype="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test[:165]"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 165 #2048
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


def preprocess(example):
    messages = [
        [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What does the image show?"}
                ]
            }
        ],
    ]
    return {
        "text": processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        ),
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return processor(sample["image"], sample["text"], add_special_tokens=False, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH)


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with GPTQ with a group size 128
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"], update_size=NUM_CALIBRATION_SAMPLES, dampening_frac=0.5)

# Apply algorithms.
oneshot(
    model=model,
    tokenizer=MODEL_ID,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
