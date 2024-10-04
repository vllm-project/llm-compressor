from transformers import AutoTokenizer

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot

MODEL_ID = "neuralmagic/SparseLlama-3-8B-pruned_50.2of4"

# Load model.
model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

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
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: true
                        symmetric: true
                    targets: ["Linear"]
    pruning_modifiers:
        ConstantPruningModifier:
            targets: [
                're:.*q_proj.weight',
                're:.*k_proj.weight', 
                're:.*v_proj.weight',
                're:.*o_proj.weight',
                're:.*gate_proj.weight',
                're:.*up_proj.weight',
                're:.*down_proj.weight',
            ]
            start: 0
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
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8.2of4"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
