from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "/raid/engine/dsikka/models--Qwen--Qwen3.5-397B-A17B/snapshots/7cad2bae11cb49ca79f7d6a0954de2e2756f4e27"

# Load model.
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with channel-wise quantization
#   * quantize the activations to fp8 with dynamic token activations
# NOTE: only datafree quantization is supported for Qwen3-VL-MoE currently
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*conv1d$",
        "re:.*in_proj_a$",
        "re:.*in_proj_b$",
        "re:.*in_proj_ba$",
        "re:.*norm$",
        "re:.*pre_fc_norm_hidden$",
        "re:.*pre_fc_norm_embedding$",
    ],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Save to disk in compressed-tensors format.
SAVE_DIR = "/raid/engine/dsikka/" + "Qwen3.5-397B-A17B" + "-FP8-Block"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
