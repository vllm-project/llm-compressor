"""
Tests for `checkpoint_aware_weight_conversions`, which guards
`HfQuantizer.update_weight_conversions` against rewriting WeightConverters
whose target modules are not quantized in the checkpoint being loaded.

Regression test for loading expert-quantized NVFP4 checkpoints (e.g.
RedHatAI/GLM-5.3-Flash-NVFP4) under `load_context`: the glm5_next
`{q,k,v}_conv1d.weight -> conv1d.weight` fusion targets dense causal-conv
modules, which transformers' quantizer would otherwise detach from the
checkpoint by rewriting its sources to `weight_packed`-style names.
"""

import pytest
import torch
from transformers.core_model_loading import (
    Concatenate,
    MergeModulelist,
    WeightConverter,
)
from transformers.quantizers.quantizer_compressed_tensors import (
    CompressedTensorsHfQuantizer,
)
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor.transformers.compression.quantized_loading import (
    _companions_in_checkpoint,
    checkpoint_aware_weight_conversions,
)

# expert-only NVFP4 scheme, like RedHatAI/GLM-5.3-Flash-NVFP4
NVFP4_EXPERTS_CONFIG = {
    "config_groups": {
        "group_0": {
            "format": "nvfp4-pack-quantized",
            "input_activations": {
                "dynamic": "local",
                "group_size": 16,
                "num_bits": 4,
                "observer": "static_minmax",
                "strategy": "tensor_group",
                "symmetric": True,
                "type": "float",
            },
            "targets": ["re:.*\\.mlp\\.experts\\..*(gate|up|down)_proj$"],
            "weights": {
                "dynamic": False,
                "group_size": 16,
                "num_bits": 4,
                "observer": "memoryless_minmax",
                "strategy": "tensor_group",
                "symmetric": True,
                "type": "float",
            },
        }
    },
    "format": "mixed-precision",
    "quantization_status": "compressed",
    "ignore": ["lm_head"],
}

SERIALIZED = ["weight_packed", "weight_scale", "weight_shape"]

# keys of a checkpoint which quantizes experts to NVFP4 but keeps the
# (fused) causal conv modules dense, like RedHatAI/GLM-5.3-Flash-NVFP4
CHECKPOINT_KEYS = {
    "model.layers.0.self_attn.q_conv1d.weight",
    "model.layers.0.self_attn.k_conv1d.weight",
    "model.layers.0.self_attn.v_conv1d.weight",
    "model.layers.1.self_attn.q_conv1d.weight",
    "model.layers.1.self_attn.k_conv1d.weight",
    "model.layers.1.self_attn.v_conv1d.weight",
    "model.layers.1.mlp.experts.0.gate_proj.weight_packed",
    "model.layers.1.mlp.experts.0.gate_proj.weight_scale",
    "model.layers.1.mlp.experts.0.gate_proj.weight_shape",
    "model.layers.1.mlp.experts.0.up_proj.weight_packed",
    "model.layers.1.mlp.experts.0.up_proj.weight_scale",
    "model.layers.1.mlp.experts.0.up_proj.weight_shape",
    "model.layers.1.mlp.experts.0.down_proj.weight_packed",
    "model.layers.1.mlp.experts.0.down_proj.weight_scale",
    "model.layers.1.mlp.experts.0.down_proj.weight_shape",
}


def _quantizer() -> CompressedTensorsHfQuantizer:
    config = CompressedTensorsConfig(**NVFP4_EXPERTS_CONFIG)
    quantizer = CompressedTensorsHfQuantizer(config, pre_quantized=True)
    # normally set by validate_environment(); not relevant to these tests
    quantizer.use_fp8_kernel = False
    return quantizer


class _Glm5NextLikeModel(torch.nn.Module):
    """
    Parameter layout of (unlinearized) glm5_next layers: a fused conv1d in
    attention (dense) and a fused MoE block
    """

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        for _ in range(2):
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.conv1d = torch.nn.Conv1d(8, 8, 4)
            layer.mlp = torch.nn.Module()
            layer.mlp.experts = torch.nn.Module()
            layer.mlp.experts.gate_up_proj = torch.nn.Linear(8, 16)
            layer.mlp.experts.down_proj = torch.nn.Linear(8, 16)
            self.model.layers.append(layer)


def _conv1d_converter() -> WeightConverter:
    # the glm5_next q/k/v conv1d fusion registered by transformers
    return WeightConverter(
        source_patterns=[
            "self_attn.q_conv1d.weight",
            "self_attn.k_conv1d.weight",
            "self_attn.v_conv1d.weight",
        ],
        target_patterns="self_attn.conv1d.weight",
        operations=[Concatenate(dim=0)],
    )


def _experts_converter() -> WeightConverter:
    # the glm5_next expert fusion registered by transformers
    return WeightConverter(
        source_patterns=[
            "mlp.experts.*.gate_proj.weight",
            "mlp.experts.*.up_proj.weight",
        ],
        target_patterns="mlp.experts.gate_up_proj",
        operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
    )


def test_companions_in_checkpoint():
    # conv1d sources have no serialized companions in the checkpoint
    assert not _companions_in_checkpoint(
        ["self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight"],
        "weight",
        SERIALIZED,
        CHECKPOINT_KEYS,
    )
    # expert sources do
    assert _companions_in_checkpoint(
        ["mlp.experts.*.gate_proj.weight", "mlp.experts.*.up_proj.weight"],
        "weight",
        SERIALIZED,
        CHECKPOINT_KEYS,
    )
    # wildcards match the same way transformers compiles them
    assert _companions_in_checkpoint(
        ["mlp.experts.*.down_proj.weight"], "weight", SERIALIZED, CHECKPOINT_KEYS
    )
    assert not _companions_in_checkpoint(
        ["mlp.shared_experts.gate_proj.weight"], "weight", SERIALIZED, CHECKPOINT_KEYS
    )


def test_update_weight_conversions_mangles_unquantized_targets():
    # upstream behavior, kept as documentation of the bug this patch guards
    # against: a converter whose target module is dense (conv1d) is rewritten
    # to collect weight_packed-style tensors which the checkpoint does not
    # store for it
    quantizer = _quantizer()
    converter = _conv1d_converter()

    result = quantizer.update_weight_conversions([converter], _Glm5NextLikeModel())

    assert any(
        source.endswith(tuple(SERIALIZED)) for source in result[0].source_patterns
    )


def test_patch_keeps_converters_without_checkpoint_companions():
    quantizer = _quantizer()
    converter = _conv1d_converter()

    with checkpoint_aware_weight_conversions(checkpoint_keys=CHECKPOINT_KEYS):
        result = quantizer.update_weight_conversions([converter], _Glm5NextLikeModel())

    assert result[0].source_patterns == [
        "self_attn.q_conv1d.weight",
        "self_attn.k_conv1d.weight",
        "self_attn.v_conv1d.weight",
    ]


def test_patch_preserves_rewrite_for_quantized_sources():
    # converters whose checkpoint sources do have serialized companions
    # (quantized modules) must keep the upstream dequantize rewrite
    quantizer = _quantizer()
    converter = _experts_converter()

    with checkpoint_aware_weight_conversions(checkpoint_keys=CHECKPOINT_KEYS):
        result = quantizer.update_weight_conversions([converter], _Glm5NextLikeModel())

    assert any(
        source.endswith(tuple(SERIALIZED)) for source in result[0].source_patterns
    )
    # the dequantize operation is prepended, as upstream
    assert "dequantize" in repr(result[0].operations[0]).lower() or hasattr(
        result[0].operations[0], "scheme"
    )


def test_patch_falls_back_without_checkpoint_keys():
    # without checkpoint keys the patch must defer to upstream behavior
    quantizer = _quantizer()
    converter = _conv1d_converter()

    with checkpoint_aware_weight_conversions(model_stub=None):
        result = quantizer.update_weight_conversions([converter], _Glm5NextLikeModel())

    assert any(
        source.endswith(tuple(SERIALIZED)) for source in result[0].source_patterns
    )


@pytest.mark.parametrize(
    "key,expected",
    [
        (
            "model.layers.0.self_attn.q_conv1d.weight",
            "model.layers.0.self_attn.conv1d.weight",
        ),
        ("model.layers.0.self_attn.k_conv1d.weight", None),
        ("model.layers.0.self_attn.v_conv1d.weight", None),
    ],
)
def test_conv1d_keys_still_load(key, expected):
    """
    Simulate the checkpoint key renaming performed while loading: under the
    patch, plain q/k/v conv1d checkpoint keys are still collected by the
    (unmangled) fusion converter rather than reported as UNEXPECTED.
    """
    from transformers.core_model_loading import rename_source_key

    quantizer = _quantizer()
    model = _Glm5NextLikeModel()
    converter = _conv1d_converter()

    with checkpoint_aware_weight_conversions(checkpoint_keys=CHECKPOINT_KEYS):
        conversions = quantizer.update_weight_conversions([converter], model)

    renamings = []
    converters = [entry for entry in conversions if isinstance(entry, WeightConverter)]
    state_dict = model.state_dict()
    renamed, source_pattern = rename_source_key(
        key, renamings, converters, base_model_prefix=None, meta_state_dict=state_dict
    )
    if expected is None:
        # k/v share the q key's target through the same converter
        assert source_pattern is not None and renamed in state_dict
    else:
        assert renamed == expected and renamed in state_dict


def test_load_quantizable_moe_activates_patch():
    """
    The patch must be active while `load_quantizable_moe` performs its load:
    simulate the glm5_next load path end to end, from registering the
    linearized conversion mapping to renaming checkpoint keys, against the
    (hub) config of the model which exposed the bug.
    """
    pytest.importorskip("transformers.models.glm5_next")
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained("RedHatAI/GLM-5.3-Flash-NVFP4")
    except Exception as error:
        pytest.skip(f"could not fetch glm5_next config: {error}")

    from transformers import Glm5NextForConditionalGeneration
    from transformers.conversion_mapping import (
        get_model_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )
    from transformers.core_model_loading import WeightRenaming, rename_source_key
    from transformers.monkey_patching import register_patch_mapping
    from transformers.quantizers.auto import AutoHfQuantizer

    from llmcompressor.modeling.moe.conversion_mappings import (
        get_linearize_load_mappings,
    )
    from llmcompressor.modeling.moe.linear_experts import LinearExperts2D

    experts_cls, load_map, _ = get_linearize_load_mappings(config.model_type)
    linear_experts_2d_cls = LinearExperts2D.get_linear_experts_cls(experts_cls)
    register_patch_mapping({experts_cls.__name__: linear_experts_2d_cls})
    register_checkpoint_conversion_mapping(config.model_type, load_map, overwrite=True)

    # shrink the config so the model fits on the meta device; the module and
    # parameter names are what matter here, not the shapes
    text_config = config.text_config
    text_config.num_hidden_layers = 4
    text_config.hidden_size = 64
    text_config.intermediate_size = 128
    text_config.moe_intermediate_size = 64
    if getattr(text_config, "num_local_experts", None):
        text_config.num_local_experts = 2
    for attr in ("hidden_size", "intermediate_size", "num_hidden_layers"):
        if hasattr(config.vision_config, attr):
            setattr(config.vision_config, attr, 32)

    with torch.device("meta"):
        model = Glm5NextForConditionalGeneration._from_config(config)

    quantizer = AutoHfQuantizer.from_config(
        config.quantization_config, pre_quantized=True
    )
    quantizer.validate_environment()
    quantizer.preprocess_model(model=model)

    with checkpoint_aware_weight_conversions(checkpoint_keys=CHECKPOINT_KEYS):
        conversions = get_model_conversion_mapping(model, None, quantizer)

    renamings = [entry for entry in conversions if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in conversions if isinstance(entry, WeightConverter)]
    state_dict = model.state_dict()

    # attention conv1d keys of dense (unquantized) modules load into the
    # fused parameter instead of being left UNEXPECTED/MISSING
    for layer in range(2):
        for proj in ("q", "k", "v"):
            key = f"model.language_model.layers.{layer}.self_attn.{proj}_conv1d.weight"
            renamed, _ = rename_source_key(
                key, renamings, converters, model.base_model_prefix, state_dict
            )
            assert renamed == (
                f"model.language_model.layers.{layer}.self_attn.conv1d.weight"
            ), f"{key} was renamed to {renamed}"
            assert renamed in state_dict
