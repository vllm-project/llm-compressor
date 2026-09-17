from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.evaluation import evaluate_kl_divergence
from llmcompressor.modifiers.quantization import QuantizationModifier

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
SAVE_DIR = "Meta-Llama-3-8B-Instruct-W4A16"

# Select calibration dataset for quantization.
DATASET_ID = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "train"

# Select number of calibration samples. 512 samples is a good place to start.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load model and tokenizer.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load calibration dataset and preprocess.
ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT)
ds = ds.filter(lambda x: len(x["text"].strip()) > 50)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
#   * quantize weights to 4 bit with group size 128
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

# Evaluate KL Divergence between base and quantized model on WikiText-2.
#
#   Both models are loaded concurrently (gpu_memory_utilization=0.45 each).
#   Hidden states are captured via a pre-allocated CUDA buffer — no disk I/O,
#   no enforce_eager required.
#
#   CLI equivalent:
#       python -m llmcompressor.evaluation.kld \
#           --base_model_id meta-llama/Meta-Llama-3-8B-Instruct \
#           --quantized_model_id ./Meta-Llama-3-8B-Instruct-W4A16 \
#           --dataset wikitext \
#           --dataset_config_name wikitext-2-raw-v1 \
#           --num_calibration_samples 512
print("\n\n========== KL DIVERGENCE EVALUATION ==============")
result = evaluate_kl_divergence(
    base_model_id=MODEL_ID,
    quantized_model_id=SAVE_DIR,
    dataset=DATASET_ID,
    dataset_config_name=DATASET_CONFIG,
    dataset_split="test",
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    max_seq_length=MAX_SEQUENCE_LENGTH,
)
print(result)
print(f"Mean KLD (base vs W4A16): {result.mean_kld:.6f}")
print("===================================================\n\n")
