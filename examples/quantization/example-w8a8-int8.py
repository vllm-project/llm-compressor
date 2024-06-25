from datasets import load_dataset
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from transformers import AutoTokenizer

# Select model and load it.
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype='auto',
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES=512
MAX_SEQUENCE_LENGTH=2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"], tokenize=False,
    )}
ds = ds.map(preprocess)

# Configure algorithms. In this case, we:
#   * apply SmoothQuant to make the activations easier to quantize
#   * quantize the weights to 8 bit with GPTQ with a static per channel strategy
#   * quantize the activations to 8 bit with a dynamic per token strategy
recipe = """
quant_stage:
    quant_modifiers:
        SmoothQuantModifier:
            smoothing_strength: 0.8
            mappings: [
                [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
                [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"]
            ]
        GPTQModifier:
            sequential_update: false
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: "int"
                        symmetric: true
                        strategy: "channel"
                    input_activations:
                        num_bits: 8
                        type: "int"
                        symmetric: true
                        dynamic: true
                        strategy: "token"
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

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-W8A8-DYNAMIC-PER-TOKEN"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
