from compressed_tensors.quantization.quant_scheme import (
    MXFP4,
    MXFP8,
    QuantizationScheme,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4PreTrainedModel,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Upstream BUG: norms should be loaded in float32, but usually aren't due to the base
# model having a quant_config which overrides this. Loading in float32 actually
# breaks the model definition (it expects bfloat16). Let's force load in bfloat16.
DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict = set()

# Select model and load it.
MODEL_ID = "RedHatAI/DeepSeek-V4-Flash-BF16"

with load_context():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu")

# kluge for the way I saved the decompressed checkpoint
# mds = model.model.layers[-1].self_attn.wq_a._hf_hook.weights_map.dataset.index
# mds["model.hc_head.base"] = mds['model.hc_head.hc_base']
# mds["model.hc_head.fn"] = mds['model.hc_head.hc_fn']
# mds["model.hc_head.scale"] = mds['model.hc_head.hc_scale']

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm to run.
#   * quantize mlp/expert weights and input activations to MXFP4
#   * quantize attention projection weights and input activations to MXFP8
# model.model.layers.0.self_attn.q_a_proj
#
# wq_a  | q_a_proj
# wq_b  | q_b_proj
# wkv   | kv_proj
# wo_a  | o_a_proj
# wo_b  | o_b_proj

recipe = QuantizationModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=[
                r"re:.*attn\.(q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)$",
                r"re:.*attn\.compressor\.indexer\.q_b_proj$",
            ],
            **MXFP8,
        ),
        "experts": QuantizationScheme(
            targets=[
                r"re:.*mlp\..*(gate|up|down)_proj$",
            ],
            **MXFP4,
        ),
    },
    ignore=[],
)

# MXFP4 and MXFP8 use RTN for weights and dynamic activation quantization, so no
# calibration dataset is required.
oneshot(
    model=model,
    recipe=recipe,
    pipeline="datafree",
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-Mixed-MXFP4-MXFP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
