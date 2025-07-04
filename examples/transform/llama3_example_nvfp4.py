import lm_eval
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import TransformModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


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

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with GPTQ with a group size 128
recipe = [
    TransformModifier(),
    QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"]),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    pipeline="sequential",
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
dispatch_for_generation(model)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# -------------------- Evals -------------------------- #
"""
# Dense
output_dense = lm_eval.simple_evaluate(model="hf", model_args={"pretrained": MODEL_ID, "add_bos_token": True}, tasks="gsm8k", batch_size=100, limit=100)
print(output_dense.get("results")["gsm8k"])

{'alias': 'gsm8k', 'exact_match,strict-match': np.float64(0.73), 'exact_match_stderr,strict-match': 0.044619604333847394, 'exact_match,flexible-extract': np.float64(0.73), 'exact_match_stderr,flexible-extract': 0.044619604333847394}

# NVFP4
output_nvfp4 = lm_eval.simple_evaluate(model="hf", model_args={"pretrained": model, "add_bos_token": True}, tasks="gsm8k", batch_size=1, limit=100)
print(output_nvfp4.get("results")["gsm8k"])

{'alias': 'gsm8k', 'exact_match,strict-match': np.float64(0.6), 'exact_match_stderr,strict-match': 0.04923659639173309, 'exact_match,flexible-extract': np.float64(0.6), 'exact_match_stderr,flexible-extract': 0.04923659639173309}
"""

# NVFP4 + Weight Transforms
output_nvfp4_transforms = lm_eval.simple_evaluate(
    model="hf",
    model_args={"pretrained": model, "add_bos_token": True},
    tasks="gsm8k",
    batch_size=1,
    limit=100,
)
print(output_nvfp4_transforms.get("results")["gsm8k"])

"""
{'alias': 'gsm8k', 'exact_match,strict-match': np.float64(0.65), 'exact_match_stderr,strict-match': 0.047937248544110196, 'exact_match,flexible-extract': np.float64(0.65), 'exact_match_stderr,flexible-extract': 0.047937248544110196}

"""

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-NVFP4-Transforms"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
