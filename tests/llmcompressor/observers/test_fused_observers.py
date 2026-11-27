"""Fused observer groups share global scales within a sequential subgraph."""

import torch
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def test_fused_observers_with_decoder_sequential_targets(tmp_path):
    model_id = "nm-testing/tinysmokellama-3.2"
    output = tmp_path / "quantized_output"
    recipe = QuantizationModifier(targets=["Linear"], scheme="NVFP4")

    # A decoder owns each complete Q/K/V and gate/up fusion group.
    oneshot(
        model=model_id,
        dataset="open_platypus",
        recipe=recipe,
        output_dir=output,
        num_calibration_samples=4,
        splits={"calibration": "train[:4]"},
        sequential_targets=["LlamaDecoderLayer"],
    )

    model = AutoModelForCausalLM.from_pretrained(output)
    layer = model.model.layers[0]
    for group in (
        [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        [layer.mlp.gate_proj, layer.mlp.up_proj],
    ):
        for module in group[1:]:
            torch.testing.assert_close(
                group[0].weight_global_scale, module.weight_global_scale
            )
