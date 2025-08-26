from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


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

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize all weights excluding down_proj layers
#       to fp4 with per group 16 via ptq
#   * calibrate a global_scale for activations, which will be used to
#       quantize activations to fp4 on the fly
#   * quantize all down_proj layer weights to fp8
#   * dynamically quantize all down_proj activations to fp8 dynamic
#       per token
recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: channel
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: token
                        dynamic: true
                        symmetric: true
                    targets: ["re:.*down_proj.*"]
                group_1:
                    weights:
                        num_bits: 4
                        type: float
                        strategy: tensor_group
                        dynamic: false
                        symmetric: true
                        group_size: 16
                    input_activations:
                        num_bits: 4
                        type: float
                        strategy: tensor_group
                        dynamic: local
                        symmetric: true
                        group_size: 16
                    targets: ["re:.*self_attn.k_proj.*", "re:.*self_attn.o_proj.*",
                        "re:.*self_attn.q_proj.*", "re:.*self_attn.v_proj.*",
                        "re:.*gate_proj.*", "re:.*up_proj.*"]
"""
# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")


# Save to disk in compressed-tensors format.

# The model produced is compressed using two different compressors
# with two different formats: nvfp4-pack-quantized and float-quantized.
# The presence of multiple compressors is indicated by the
# `mixed-precision` format in the model's config.json.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-FP8-Dynamic"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
