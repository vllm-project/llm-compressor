import os
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

# BUG in norms which is masked by quant config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4PreTrainedModel
DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict = {
    #"attn_hc",
    #"ffn_hc",
    #"e_score_correction_bias",
    #"q_a_norm",
    #"kv_norm",
    #"input_layernorm",
    #"post_attention_layernorm",
    #"norm",
}

from llmcompressor.utils.dev import skip_weights_download
from llmcompressor.modeling.moe.linearize import load_linearized_moe


def expert_key_exists(model_path: Path, key: str) -> bool:
    """
    Utility to check that expected expert keys exist in a saved model.

    Args:
        model_path: Path to the saved model directory
        expected_patterns: List of key patterns to check for

    Returns:
        True if all expected patterns are found in the model checkpoint
    """
    safetensor_files = list(model_path.glob("*.safetensors"))
    all_keys = set()

    for st_file in safetensor_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            all_keys.update(f.keys())

    return key in all_keys


def test_linearize_moe_model(tmp_path):
    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    input_ids = torch.randint(1024, size=(1, 64), device="cuda")
    model = AutoModelForCausalLM.from_pretrained(
        "inference-optimization/DSV4-tiny-empty", device_map="cuda",
    )
    true_outputs = model(input_ids=input_ids).logits
    del model

    # TODO: revert/inverse conversion seems to not work for this model
    # as the checkpoint contains model.layers.0.mlp.experts.gate_up_proj, not model.layers.0.ffn.experts.0.w1

    # proper weight converter is still on gotten by model before inverting
    # WeightConverter(source_patterns=['experts.*.w1.weight', 'experts.*.w3.weight'], ta
    # rget_patterns=['experts.gate_up_proj']), WeightConverter(source_patterns=['experts.*.w2.weight'], target_patterns=['expert
    # s.down_proj'])

    with load_linearized_moe():
        linearized_model = AutoModelForCausalLM.from_pretrained("inference-optimization/DSV4-tiny-empty", device_map="cuda")

    breakpoint()

    # breakpoint()

    # # check calibrate_all_experts=True
    # with ...:
    #     linearized_model = model(*input_ids)
    #     assert linearized_model == true_outputs

    # # check calibrate_all_experts=False
    # with ...:
    #     linearized_model = model(*input_ids)
    #     assert linearized_model == true_outputs

    # linearized_model.save_pretrained(model)
    # # check checkpoint keys for experts.N.up_proj structure
