import pytest
import torch
from torch.nn import Linear

from llmcompressor.modifiers.transform.awq.dynamic_mappings import (
    AWQ_DYNAMIC_MAPPING_REGISTRY,
    _detect_linear_attn_projections,
    _get_hybrid_attention_config,
    build_hybrid_attention_mappings,
    get_layer_mappings_from_model,
)


def _make_hybrid_model(
    num_layers=4,
    full_attention_interval=4,
    linear_proj_names=("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
    moe=False,
    num_experts=2,
    use_text_config=False,
):
    """Build a minimal hybrid attention model for testing."""
    layer_types = [
        (
            "full_attention"
            if i % full_attention_interval == full_attention_interval - 1
            else "linear_attention"
        )
        for i in range(num_layers)
    ]

    layers = []
    for i in range(num_layers):
        if layer_types[i] == "full_attention":
            attn = torch.nn.ModuleDict(
                {
                    "q_proj": Linear(8, 8),
                    "k_proj": Linear(8, 8),
                    "v_proj": Linear(8, 8),
                    "o_proj": Linear(8, 8),
                }
            )
            layer = torch.nn.ModuleDict({"self_attn": attn})
        else:
            attn = torch.nn.ModuleDict(
                {name: Linear(8, 8) for name in linear_proj_names}
            )
            attn["norm"] = torch.nn.LayerNorm(8)
            attn["out_proj"] = Linear(8, 8)
            layer = torch.nn.ModuleDict({"linear_attn": attn})

        if moe:
            experts = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "gate_proj": Linear(8, 8),
                            "up_proj": Linear(8, 8),
                            "down_proj": Linear(8, 8),
                        }
                    )
                    for _ in range(num_experts)
                ]
            )
            shared = torch.nn.ModuleDict(
                {
                    "gate_proj": Linear(8, 8),
                    "up_proj": Linear(8, 8),
                    "down_proj": Linear(8, 8),
                }
            )
            layer["mlp"] = torch.nn.ModuleDict(
                {
                    "experts": experts,
                    "shared_expert": shared,
                }
            )
        else:
            layer["mlp"] = torch.nn.ModuleDict(
                {
                    "gate_proj": Linear(8, 8),
                    "up_proj": Linear(8, 8),
                    "down_proj": Linear(8, 8),
                }
            )

        layer["input_layernorm"] = torch.nn.LayerNorm(8)
        layer["post_attention_layernorm"] = torch.nn.LayerNorm(8)
        layers.append(layer)

    model = torch.nn.ModuleDict(
        {
            "model": torch.nn.ModuleDict(
                {
                    "layers": torch.nn.ModuleList(layers),
                }
            )
        }
    )

    # Attach a config
    config_attrs = {
        "num_hidden_layers": num_layers,
        "layer_types": layer_types,
    }
    if moe:
        config_attrs["num_local_experts"] = num_experts
    config = type("Config", (), config_attrs)()

    if use_text_config:
        model.config = type("Config", (), {"text_config": config})()
    else:
        model.config = config

    return model


def _make_standard_model():
    """Build a minimal standard (non-hybrid) attention model."""
    layers = []
    for _ in range(4):
        layer = torch.nn.ModuleDict(
            {
                "self_attn": torch.nn.ModuleDict(
                    {
                        "q_proj": Linear(8, 8),
                        "k_proj": Linear(8, 8),
                        "v_proj": Linear(8, 8),
                        "o_proj": Linear(8, 8),
                    }
                ),
                "mlp": torch.nn.ModuleDict(
                    {
                        "gate_proj": Linear(8, 8),
                        "up_proj": Linear(8, 8),
                        "down_proj": Linear(8, 8),
                    }
                ),
                "input_layernorm": torch.nn.LayerNorm(8),
                "post_attention_layernorm": torch.nn.LayerNorm(8),
            }
        )
        layers.append(layer)

    model = torch.nn.ModuleDict(
        {
            "model": torch.nn.ModuleDict(
                {
                    "layers": torch.nn.ModuleList(layers),
                }
            )
        }
    )
    # No layer_types in config = not hybrid
    model.config = type("Config", (), {"num_hidden_layers": 4})()
    return model


@pytest.mark.unit
class TestGetHybridAttentionConfig:
    def test_returns_config_for_hybrid_model(self):
        model = _make_hybrid_model(num_layers=8)
        result = _get_hybrid_attention_config(model)
        assert result is not None
        layer_types, num_layers = result
        assert num_layers == 8
        assert layer_types.count("full_attention") == 2
        assert layer_types.count("linear_attention") == 6

    def test_returns_none_for_standard_model(self):
        model = _make_standard_model()
        assert _get_hybrid_attention_config(model) is None

    def test_reads_text_config_for_vl_models(self):
        model = _make_hybrid_model(num_layers=4, use_text_config=True)
        result = _get_hybrid_attention_config(model)
        assert result is not None
        _, num_layers = result
        assert num_layers == 4

    def test_returns_none_without_config(self):
        model = torch.nn.Linear(4, 4)
        assert _get_hybrid_attention_config(model) is None


@pytest.mark.unit
class TestDetectLinearAttnProjections:
    def test_qwen3_5_projections(self):
        model = _make_hybrid_model(
            linear_proj_names=("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        )
        projs = _detect_linear_attn_projections(model)
        assert projs == ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]

    def test_qwen3next_projections(self):
        model = _make_hybrid_model(linear_proj_names=("in_proj_qkvz", "in_proj_ba"))
        projs = _detect_linear_attn_projections(model)
        assert projs == ["in_proj_qkvz", "in_proj_ba"]

    def test_deduplicates_across_layers(self):
        model = _make_hybrid_model(num_layers=8)
        projs = _detect_linear_attn_projections(model)
        # 6 linear layers but should only return unique projection names
        assert len(projs) == len(set(projs))


@pytest.mark.unit
class TestMoeDetectionInMappings:
    def test_moe_model_gets_expert_mlp_mappings(self):
        model = _make_hybrid_model(moe=True, num_experts=4)
        mappings = build_hybrid_attention_mappings(model)
        assert mappings is not None
        mlp_mapping = mappings[2]
        assert any("experts" in b for b in mlp_mapping.balance_layers)

    def test_dense_model_gets_simple_mlp_mappings(self):
        model = _make_hybrid_model(moe=False)
        mappings = build_hybrid_attention_mappings(model)
        assert mappings is not None
        mlp_mapping = mappings[2]
        assert not any("experts" in b for b in mlp_mapping.balance_layers)


@pytest.mark.unit
class TestBuildHybridAttentionMappings:
    def test_qwen3_5_dense(self):
        """Qwen3.5-style: 8 layers, dense MLP, 4 separate linear projections."""
        model = _make_hybrid_model(
            num_layers=8,
            linear_proj_names=("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
            moe=False,
        )
        mappings = build_hybrid_attention_mappings(model)
        assert mappings is not None
        assert len(mappings) == 4

        # Full attention: layers 3, 7
        full_attn_mapping = mappings[0]
        assert "3|7" in full_attn_mapping.smooth_layer
        assert len(full_attn_mapping.balance_layers) == 3  # q, k, v

        # Linear attention: layers 0,1,2,4,5,6
        linear_mapping = mappings[1]
        assert "0|1|2|4|5|6" in linear_mapping.smooth_layer
        assert len(linear_mapping.balance_layers) == 4  # qkv, z, b, a

        # MLP: dense
        mlp_mapping = mappings[2]
        assert any("gate_proj" in b for b in mlp_mapping.balance_layers)
        assert not any("experts" in b for b in mlp_mapping.balance_layers)

    def test_qwen3next_moe(self):
        """Qwen3Next-style: MoE, 2 fused linear projections."""
        model = _make_hybrid_model(
            num_layers=8,
            linear_proj_names=("in_proj_qkvz", "in_proj_ba"),
            moe=True,
            num_experts=4,
        )
        mappings = build_hybrid_attention_mappings(model)
        assert mappings is not None

        linear_mapping = mappings[1]
        assert len(linear_mapping.balance_layers) == 2  # qkvz, ba

        mlp_mapping = mappings[2]
        assert any("experts" in b for b in mlp_mapping.balance_layers)
        assert any("shared_expert" in b for b in mlp_mapping.balance_layers)

    def test_returns_none_for_standard_model(self):
        model = _make_standard_model()
        assert build_hybrid_attention_mappings(model) is None

    def test_layer_indices_scale_with_model_size(self):
        """Verify dynamic indices work for different layer counts."""
        for num_layers in (24, 48, 64):
            model = _make_hybrid_model(num_layers=num_layers)
            mappings = build_hybrid_attention_mappings(model)
            assert mappings is not None

            full_re = mappings[0].smooth_layer
            linear_re = mappings[1].smooth_layer

            # Count indices in the regex
            full_count = full_re.count("|") + 1
            linear_count = linear_re.count("|") + 1

            expected_full = num_layers // 4
            expected_linear = num_layers - expected_full
            assert full_count == expected_full
            assert linear_count == expected_linear


@pytest.mark.unit
class TestGetLayerMappingsFromModel:
    def test_dynamic_registry_model_uses_dynamic_path(self):
        model = _make_hybrid_model(num_layers=8)
        # Fake the class name to match a dynamic registry entry
        model.__class__ = type(
            "Qwen3_5ForConditionalGeneration", (model.__class__,), {}
        )
        assert model.__class__.__name__ in AWQ_DYNAMIC_MAPPING_REGISTRY
        mappings = get_layer_mappings_from_model(model)
        assert len(mappings) == 4
        assert any("|" in m.smooth_layer for m in mappings)

    def test_static_registry_model_uses_static_path(self):
        model = _make_standard_model()
        model.__class__ = type("LlamaForCausalLM", (model.__class__,), {})
        mappings = get_layer_mappings_from_model(model)
        assert len(mappings) == 4
        assert not any("|" in m.smooth_layer for m in mappings)

    def test_unknown_model_gets_default_mappings(self):
        model = _make_standard_model()
        model.__class__ = type("SomeNewModelNobodyKnows", (model.__class__,), {})
        mappings = get_layer_mappings_from_model(model)
        assert len(mappings) == 4
        assert not any("|" in m.smooth_layer for m in mappings)

    def test_vl_model_reads_text_config(self):
        model = _make_hybrid_model(num_layers=4, use_text_config=True)
        model.__class__ = type(
            "Qwen3_5ForConditionalGeneration", (model.__class__,), {}
        )
        mappings = get_layer_mappings_from_model(model)
        assert mappings is not None
        assert len(mappings) == 4
