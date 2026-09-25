import pytest
import torch
from torch.nn import Embedding, Linear

from llmcompressor.modifiers.transform.spinquant.dynamic_mappings import (
    NORM_DYNAMIC_MAPPING_REGISTRY,
    SPINQUANT_DYNAMIC_MAPPING_REGISTRY,
    build_qwen3_5_norm_mappings,
    build_qwen3_5_spinquant_mapping,
)
from llmcompressor.modifiers.transform.spinquant.mappings import (
    SPINQUANT_MAPPING_REGISTRY,
    infer_mapping_from_model,
)
from llmcompressor.modifiers.transform.spinquant.norm_mappings import (
    NORM_MAPPING_REGISTRY,
    infer_norm_mapping_from_model,
)


def _make_qwen3_5_model(
    num_layers=4,
    full_attention_interval=4,
    linear_proj_names=("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
):
    """Build a minimal Qwen3.5-shaped model (language_model + lm_head)."""
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

    language_model = torch.nn.ModuleDict(
        {
            "embed_tokens": Embedding(16, 8),
            "layers": torch.nn.ModuleList(layers),
            "norm": torch.nn.LayerNorm(8),
        }
    )
    model = torch.nn.ModuleDict(
        {
            "language_model": language_model,
            "lm_head": Linear(8, 16),
        }
    )

    config = type(
        "Config",
        (),
        {"num_hidden_layers": num_layers, "layer_types": layer_types},
    )()
    model.config = type("Config", (), {"text_config": config})()
    return model


def _make_standard_model():
    """Build a minimal standard (non-hybrid) Llama-shaped model."""
    layers = []
    for _ in range(4):
        layers.append(
            torch.nn.ModuleDict(
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
        )

    model = torch.nn.ModuleDict(
        {
            "model": torch.nn.ModuleDict(
                {
                    "embed_tokens": Embedding(16, 8),
                    "layers": torch.nn.ModuleList(layers),
                    "norm": torch.nn.LayerNorm(8),
                }
            ),
            "lm_head": Linear(8, 16),
        }
    )
    model.config = type("Config", (), {"num_hidden_layers": 4})()
    return model


@pytest.mark.unit
class TestBuildQwen35SpinquantMapping:
    def test_returns_none_for_standard_model(self):
        model = _make_standard_model()
        assert build_qwen3_5_spinquant_mapping(model) is None

    def test_folds_linear_attn_projections_into_mlp(self):
        model = _make_qwen3_5_model(num_layers=8)
        mapping = build_qwen3_5_spinquant_mapping(model)
        assert mapping is not None

        # linear attention input projections are read by R1's inverse rotation
        for proj in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
            assert any(proj in target for target in mapping.mlp_in)
        # linear attention output projection writes the residual stream
        assert any("out_proj" in target for target in mapping.mlp_out)
        # MLP projections still present
        assert any("up_proj" in target for target in mapping.mlp_in)
        assert any("gate_proj" in target for target in mapping.mlp_in)
        assert any("down_proj" in target for target in mapping.mlp_out)

    def test_embedding_and_lm_head(self):
        model = _make_qwen3_5_model(num_layers=8)
        mapping = build_qwen3_5_spinquant_mapping(model)
        assert mapping.embedding == "re:.*embed_tokens$"
        assert mapping.lm_head == "lm_head"


@pytest.mark.unit
class TestBuildQwen35NormMappings:
    def test_returns_none_for_standard_model(self):
        model = _make_standard_model()
        assert build_qwen3_5_norm_mappings(model) is None

    def test_layer_index_restricted_input_layernorm(self):
        model = _make_qwen3_5_model(num_layers=8)
        mappings = build_qwen3_5_norm_mappings(model)
        assert mappings is not None
        assert len(mappings) == 4

        full_norm = mappings[0]
        assert "3|7" in full_norm.norm
        assert full_norm.linears == [
            "re:.*self_attn\\.q_proj$",
            "re:.*self_attn\\.k_proj$",
            "re:.*self_attn\\.v_proj$",
        ]

        linear_norm = mappings[1]
        assert "0|1|2|4|5|6" in linear_norm.norm
        assert len(linear_norm.linears) == 4

        post_norm = mappings[2]
        assert post_norm.norm == "re:.*post_attention_layernorm$"

        final_norm = mappings[3]
        assert final_norm.norm == "language_model.norm"
        assert final_norm.linears == ["lm_head"]

    def test_each_group_has_single_norm(self):
        from compressed_tensors import match_modules_set

        model = _make_qwen3_5_model(num_layers=8)
        mappings = build_qwen3_5_norm_mappings(model)
        for mapping in mappings:
            for norm, *linears in match_modules_set(
                model, (mapping.norm, *mapping.linears)
            ):
                assert len(norm) == 1


@pytest.mark.unit
class TestGetMappingsFromModel:
    def test_qwen3_5_uses_dynamic_path(self):
        model = _make_qwen3_5_model(num_layers=8)
        model.__class__ = type(
            "Qwen3_5ForConditionalGeneration", (model.__class__,), {}
        )
        assert model.__class__.__name__ in SPINQUANT_DYNAMIC_MAPPING_REGISTRY
        assert model.__class__.__name__ in NORM_DYNAMIC_MAPPING_REGISTRY

        mapping = infer_mapping_from_model(model)
        assert any("in_proj_qkv" in t for t in mapping.mlp_in)

        norm_mappings = infer_norm_mapping_from_model(model)
        assert len(norm_mappings) == 4

    def test_llama_uses_static_path(self):
        model = _make_standard_model()
        model.__class__ = type("LlamaForCausalLM", (model.__class__,), {})
        assert (
            infer_mapping_from_model(model)
            is SPINQUANT_MAPPING_REGISTRY["LlamaForCausalLM"]
        )
        assert (
            infer_norm_mapping_from_model(model)
            is NORM_MAPPING_REGISTRY["LlamaForCausalLM"]
        )

    def test_unknown_uses_defaults(self):
        model = _make_standard_model()
        model.__class__ = type("SomeNewModelNobodyKnows", (model.__class__,), {})
        mapping = infer_mapping_from_model(model)
        assert mapping.embedding == "re:.*embed_tokens$"
        norm_mappings = infer_norm_mapping_from_model(model)
        assert len(norm_mappings) == 3
