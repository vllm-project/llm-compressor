from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llmcompressor.modifiers.transform.smoothquant.dynamic_mappings import (
    SMOOTHQUANT_DYNAMIC_MAPPING_REGISTRY,
    build_qwen3_5_dense_smoothquant_mappings,
    build_qwen3_5_moe_smoothquant_mappings,
    get_layer_mappings_from_model,
)
from llmcompressor.modifiers.transform.smoothquant.utils import (
    COHERE_SMOOTHQUANT_MAPPINGS,
    DEEPSEEK_V2_SMOOTHQUANT_MAPPINGS,
    DEFAULT_SMOOTHQUANT_MAPPINGS,
    MAPPINGS_REGISTRY,
    PHI3_VISION_SMOOTHQUANT_MAPPINGS,
    get_layer_mappings_from_architecture,
    handle_mapping_resolution_errors,
)

smoothquant_dynamic_mappings = (
    "llmcompressor.modifiers.transform.smoothquant.dynamic_mappings"
)
smoothquant_utils = "llmcompressor.modifiers.transform.smoothquant.utils"


@pytest.mark.unit
def test_handle_mapping_resolution_errors():
    README_LOCATION = (
        "https://github.com/vllm-project/llm-compressor/tree/main/"
        "src/llmcompressor/modifiers/transform/smoothquant"
    )

    @handle_mapping_resolution_errors
    def func_that_raises_exception():
        raise ValueError("An error occurred")

    with pytest.raises(RuntimeError) as excinfo:
        func_that_raises_exception()

    assert "Error resolving mappings for given architecture." in str(excinfo.value)
    assert "Please refer to the README at" in str(excinfo.value)
    assert README_LOCATION in str(excinfo.value)


@pytest.mark.unit
@patch(
    f"{smoothquant_utils}.MAPPINGS_REGISTRY", {"arch1": "mapping1", "arch2": "mapping2"}
)
@patch(f"{smoothquant_utils}.DEFAULT_SMOOTHQUANT_MAPPINGS", "default_mapping")
def test_get_layer_mappings_from_architecture():
    # Test when architecture is in MAPPINGS_REGISTRY
    assert get_layer_mappings_from_architecture("arch1") == "mapping1"

    # Test when architecture is not in MAPPINGS_REGISTRY
    assert get_layer_mappings_from_architecture("arch3") == "default_mapping"


@pytest.mark.unit
@pytest.mark.parametrize(
    "architecture,expected_mappings",
    [
        ("CohereForCausalLM", COHERE_SMOOTHQUANT_MAPPINGS),
        ("Cohere2ForCausalLM", COHERE_SMOOTHQUANT_MAPPINGS),
        ("Cohere2VisionForConditionalGeneration", COHERE_SMOOTHQUANT_MAPPINGS),
    ],
)
def test_new_architecture_mappings_resolve(architecture, expected_mappings):
    assert architecture in MAPPINGS_REGISTRY
    resolved = get_layer_mappings_from_architecture(architecture)
    assert resolved is expected_mappings
    assert resolved is not DEFAULT_SMOOTHQUANT_MAPPINGS


@pytest.mark.unit
@pytest.mark.parametrize(
    "architecture,expected_mappings",
    [
        ("DeepseekV3ForCausalLM", DEEPSEEK_V2_SMOOTHQUANT_MAPPINGS),
        ("Phi3ForCausalLM", PHI3_VISION_SMOOTHQUANT_MAPPINGS),
    ],
)
def test_specialized_architecture_overrides_default(architecture, expected_mappings):
    assert architecture in MAPPINGS_REGISTRY
    resolved = get_layer_mappings_from_architecture(architecture)
    assert resolved is expected_mappings
    assert resolved is not DEFAULT_SMOOTHQUANT_MAPPINGS


@pytest.mark.unit
def test_build_qwen3_5_moe_smoothquant_mappings_uses_text_config_layer_types():
    model = type("Qwen3_5MoeForConditionalGeneration", (), {})()
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(
            layer_types=[
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ]
        )
    )

    mappings = build_qwen3_5_moe_smoothquant_mappings(model)

    assert mappings[0].smooth_layers == "re:.*layers\\.(1|3)\\.input_layernorm$"
    assert mappings[0].balance_layers == [
        "re:.*self_attn\\.q_proj$",
        "re:.*self_attn\\.k_proj$",
        "re:.*self_attn\\.v_proj$",
    ]
    assert mappings[1].smooth_layers == "re:.*post_attention_layernorm$"
    assert mappings[1].balance_layers == [
        "re:.*mlp\\.shared_expert\\.gate_proj$",
        "re:.*mlp\\.shared_expert\\.up_proj$",
    ]


@pytest.mark.unit
def test_build_qwen3_5_dense_smoothquant_mappings_uses_dense_mlp_layers():
    model = type("Qwen3_5ForCausalLM", (), {})()
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(
            layer_types=[
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ]
        )
    )

    mappings = build_qwen3_5_dense_smoothquant_mappings(model)

    assert mappings[0].smooth_layers == "re:.*layers\\.(1|3)\\.input_layernorm$"
    assert mappings[1].smooth_layers == "re:.*post_attention_layernorm$"
    assert mappings[1].balance_layers == [
        "re:.*mlp\\.gate_proj$",
        "re:.*mlp\\.up_proj$",
    ]


@pytest.mark.unit
def test_build_qwen3_5_moe_smoothquant_mappings_requires_layer_types():
    model = type("Qwen3_5MoeForConditionalGeneration", (), {})()
    model.config = SimpleNamespace(text_config=SimpleNamespace(layer_types=None))

    with pytest.raises(ValueError, match="layer_types"):
        build_qwen3_5_moe_smoothquant_mappings(model)


@pytest.mark.unit
def test_get_layer_mappings_from_model_uses_dynamic_registry():
    model = type("Qwen3_5MoeForConditionalGeneration", (), {})()
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(layer_types=["full_attention"])
    )

    mappings = get_layer_mappings_from_model(model)

    assert model.__class__.__name__ in SMOOTHQUANT_DYNAMIC_MAPPING_REGISTRY
    assert mappings[0].smooth_layers == "re:.*layers\\.(0)\\.input_layernorm$"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("architecture", "expected_balance_layers"),
    [
        (
            "Qwen3_5ForCausalLM",
            ["re:.*mlp\\.gate_proj$", "re:.*mlp\\.up_proj$"],
        ),
        (
            "Qwen3_5ForConditionalGeneration",
            ["re:.*mlp\\.gate_proj$", "re:.*mlp\\.up_proj$"],
        ),
        (
            "Qwen3_5MoeForCausalLM",
            [
                "re:.*mlp\\.shared_expert\\.gate_proj$",
                "re:.*mlp\\.shared_expert\\.up_proj$",
            ],
        ),
        (
            "Qwen3_5MoeForConditionalGeneration",
            [
                "re:.*mlp\\.shared_expert\\.gate_proj$",
                "re:.*mlp\\.shared_expert\\.up_proj$",
            ],
        ),
    ],
)
def test_qwen3_5_architectures_use_dynamic_registry(
    architecture, expected_balance_layers
):
    model = type(architecture, (), {})()
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(layer_types=["linear_attention", "full_attention"])
    )

    mappings = get_layer_mappings_from_model(model)

    assert architecture in SMOOTHQUANT_DYNAMIC_MAPPING_REGISTRY
    assert mappings[0].smooth_layers == "re:.*layers\\.(1)\\.input_layernorm$"
    assert mappings[1].balance_layers == expected_balance_layers


@pytest.mark.unit
@patch(
    f"{smoothquant_dynamic_mappings}.MAPPINGS_REGISTRY",
    {"arch1": "mapping1", "arch2": "mapping2"},
)
@patch(
    f"{smoothquant_dynamic_mappings}.DEFAULT_SMOOTHQUANT_MAPPINGS", "default_mapping"
)
@patch(f"{smoothquant_dynamic_mappings}.SMOOTHQUANT_DYNAMIC_MAPPING_REGISTRY", {})
def test_get_layer_mappings_from_model_falls_back_to_static_and_default():
    arch1 = type("arch1", (), {})()
    arch3 = type("arch3", (), {})()

    assert get_layer_mappings_from_model(arch1) == "mapping1"
    assert get_layer_mappings_from_model(arch3) == "default_mapping"
