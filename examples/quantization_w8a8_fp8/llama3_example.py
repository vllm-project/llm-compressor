from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Llama 3 has rop embeddings

# Load model.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# If not callable, would have to be a registry of reigstered callables
# replace None with Callable 
# Need to add the ability to ignore certain layers when defining "Linear"/larger groups
"""
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
            transforms:
                group_0:
                    targets: ["Linear"]
                    ignore: ["lm_head"] 
                    weights: 
                        transform_0:
                            type: hadamard
                            transpose: true
                            right_left: right
                            global: true
                        transform_1:
                            type: identity
                            transpoe: false
                    input_activations: 
                        transform_0:
                            type: identity  
                group_1:
                    targets: ["Embedding]
                    output_activations:
                        transform_0:
                            type: identity
                group_2:
                    targets: ["model.layer.21]
                    output_activations:
                        transform_0:
                            type: identity
"""                  


transforms = {
    "Linear": {
        "weight": None,
        "input_activations": None
        "ignore": []
    },
    "Embedding": {
        "output_activations": None
    },
    "model.layers.21": {
        "output_activations": None
    }
}
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8", ignore=["lm_head"], transforms=transforms
)

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 1
MAX_SEQUENCE_LENGTH = 2048

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


# Apply quantization.
oneshot(
    model=model, 
    recipe=recipe,     
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,)

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
