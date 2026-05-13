import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
from safetensors import safe_open
from llmcompressor.utils.dev import skip_weights_download
from llmcompressor.modeling.moe.linearize import load_linearized_moe


def check_expert_keys_exist(model_path: Path, expected_patterns: list[str]) -> bool:
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

    for pattern in expected_patterns:
        if not any(pattern in key for key in all_keys):
            return False

    return True

def test_linearize_moe_model(tmp_path):
    offload_dir = tmp_path / "offload_dir"
    os.mkdir(offload_dir)

    input_ids = torch.randint(1024, size=(1, 64), device="cuda")

    with skip_weights_download(skip_init=False):  # TODO: test with cpu offloading
        config = AutoConfig.from_pretrained(
            "deepseek-ai/DeepSeek-V4-Flash",
            num_hidden_layers=3,
            num_nextn_predict_layers=1,
            layer_types=["heavily_compressed_attention", "compressed_sparse_attention", "sliding_attention"],
            mlp_layer_types=["hash_moe", "moe", "moe"],
        )
        delattr(config, "quantization_config")
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V4-Flash",
            config=config,
            device_map="cuda",
        )

    true_outputs = model(input_ids=input_ids).logits
    breakpoint()
    model.save_pretrained(offload_dir)
    assert check_expert_keys_exist(offload_dir, "model.layers.0.ffn.experts.0.w1")
    del model

    # CURRENT: `check_expert_keys_exist` somehow passes, but manually inspection seemed not to show that?

    # with load_linearized_moe():
    #     linearized_model = AutoModelForCausalLM.from_pretrained(offload_dir, device_map="cuda")

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
