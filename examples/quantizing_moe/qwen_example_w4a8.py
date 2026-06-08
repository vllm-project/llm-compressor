import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

# select a Mixture of Experts model for quantization
MODEL_ID = "/mnt/exp/moe-lite/deze/test/ling_max_26_bf16_final/"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True
)


recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head", "re:.*mlp.gate$"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: block
                        dynamic: false
                        symmetric: true
                        block_structure: [128, 128]
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: token
                        dynamic: true
                        symmetric: true
                    targets:
                        - "re:.*attention.q_a_proj"
                        - "re:.*attention.q_b_proj"
                        - "re:.*attention.kv_a_proj_with_mqa"
                        - "re:.*attention.kv_b_proj"
                group_1:
                    weights:
                        num_bits: 4
                        type: int
                        strategy: tensor_group
                        dynamic: false
                        symmetric: true
                        group_size: 128
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: token
                        dynamic: true
                        symmetric: true
                    targets: [ "re:.*gate_proj", "re:.*up_proj", "re:.*down_proj"]
"""

oneshot(
    model=model,
    dataset=None,
    recipe=recipe,
    num_calibration_samples=0,
)


# Save to disk in compressed-tensors format.
model.name_or_path = MODEL_ID

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-wInt4aFp8"
model.save_pretrained(SAVE_DIR, save_compressed=True)


print("========== SAVED COMPRESSED MODEL TO DISK ==============")
