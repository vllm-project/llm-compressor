from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "Qwen/Qwen3-32B"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the importance-aware mixed-precision quantization recipe.
# In this case, we:
#   * Keep the first 3 layers (0,1,2) and last 3 layers (51,62,63) in full precision
#       (not quantized) due to their high sensitivity.
#   * Exclude lm_head from quantization to preserve output quality.
#   * Quantize weights of specific middle layers' self-attention and MLP blocks to fp4:
#       - Layers: 15-24, 31, 46-48, 50, 56-60
#       - Modules: k_proj, o_proj, q_proj, v_proj, down_proj, gate_proj, up_proj
#       - Scheme: fp4, symmetric, per-group (group_size=16), static (PTQ)
#   * Quantize weights of other intermediate layers to fp8:
#       - Layers: 3-14, 25-30, 32-45, 49, 52-55, 61
#       - Same modules as above
#       - Scheme: fp8, symmetric, per-channel, static (PTQ)
#   * Additionally, dynamically quantize input activations for fp8-weighted layers:
#       - Activations quantized to fp8, symmetric, per-token, dynamic range

# Define layer groups for readability and line-length compliance

fp4_group = "15|16|17|18|19|20|21|22|23|24|31|46|47|48|50|56|57|58|59|60"
fp8_group = (
    "3|4|5|6|7|8|9|10|11|12|13|14|25|26|27|28|29|30|"
    "32|33|34|35|36|37|38|39|40|41|42|43|44|45|49|52|53|54|55|61"
)

recipe = f"""
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore:
                - "lm_head"
                - 're:.*layers\\.0\\..*'
                - 're:.*layers\\.1\\..*'
                - 're:.*layers\\.2\\..*'
                - 're:.*layers\\.51\\..*'
                - 're:.*layers\\.62\\..*'
                - 're:.*layers\\.63\\..*'
            config_groups:
                group_0:
                    weights:
                        num_bits: 4
                        type: float
                        strategy: tensor_group
                        dynamic: false
                        symmetric: true
                        group_size: 16
                    targets:
                        - 're:.*layers\\.({fp4_group})\\.self_attn\\.[kqov]_proj'
                        - 're:.*layers\\.({fp4_group})\\.mlp\\.(gate|up|down)_proj'
                group_1:
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
                    targets:
                        - 're:.*layers\\.({fp8_group})\\.self_attn\\.[kqov]_proj'
                        - 're:.*layers\\.({fp8_group})\\.mlp\\.(gate|up|down)_proj'
"""

# Apply quantization.
oneshot(model=model, recipe=recipe)

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
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-Importance-Aware-Mix-Quantization"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
