import pytest
import torch
from compressed_tensors.quantization import apply_quantization_config
from compressed_tensors.utils import match_modules_set
from torch.nn import Linear

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.transform.awq.dynamic_mappings import (
    AWQ_DYNAMIC_MAPPING_REGISTRY,
    _detect_linear_attn_projections,
    _detect_step3p5_ffn_layer_indices,
    build_hybrid_attention_mappings,
    build_step3p5_mappings,
    get_layer_mappings_from_model,
)
from llmcompressor.modifiers.transform.awq.mappings import (
    AWQ_MAPPING_REGISTRY,
    _whisper_mappings,
    default_mappings,
)
from llmcompressor.modifiers.transform.utils.hybrid_attention import (
    get_hybrid_attention_config,
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


def _make_step3p5_model(num_layers=6, moe_start=2):
    """Build a minimal Step3p5-shaped model with dense and MoE FFN layers."""
    layers = []
    for i in range(num_layers):
        layer = torch.nn.ModuleDict(
            {
                "self_attn": torch.nn.ModuleDict(
                    {
                        "q_proj": Linear(8, 8),
                        "k_proj": Linear(8, 8),
                        "v_proj": Linear(8, 8),
                        "g_proj": Linear(8, 8),
                        "o_proj": Linear(8, 8),
                    }
                ),
                "input_layernorm": torch.nn.LayerNorm(8),
                "post_attention_layernorm": torch.nn.LayerNorm(8),
            }
        )

        if i < moe_start:
            layer["mlp"] = torch.nn.ModuleDict(
                {
                    "gate_proj": Linear(8, 8),
                    "up_proj": Linear(8, 8),
                    "down_proj": Linear(8, 8),
                }
            )
        else:
            layer["moe"] = torch.nn.ModuleDict(
                {
                    "gate": Linear(8, 8),
                    "gate_proj": Linear(8, 8),
                    "up_proj": Linear(8, 8),
                    "down_proj": Linear(8, 8),
                }
            )
            layer["share_expert"] = torch.nn.ModuleDict(
                {
                    "gate_proj": Linear(8, 8),
                    "up_proj": Linear(8, 8),
                    "down_proj": Linear(8, 8),
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
    model.config = type("Config", (), {"num_hidden_layers": num_layers})()
    return model


@pytest.mark.unit
class TestGetHybridAttentionConfig:
    def test_returns_config_for_hybrid_model(self):
        model = _make_hybrid_model(num_layers=8)
        result = get_hybrid_attention_config(model)
        assert result is not None
        layer_types, num_layers = result
        assert num_layers == 8
        assert layer_types.count("full_attention") == 2
        assert layer_types.count("linear_attention") == 6

    def test_returns_none_for_standard_model(self):
        model = _make_standard_model()
        assert get_hybrid_attention_config(model) is None

    def test_reads_text_config_for_vl_models(self):
        model = _make_hybrid_model(num_layers=4, use_text_config=True)
        result = get_hybrid_attention_config(model)
        assert result is not None
        _, num_layers = result
        assert num_layers == 4

    def test_returns_none_without_config(self):
        model = torch.nn.Linear(4, 4)
        assert get_hybrid_attention_config(model) is None


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
class TestStep3p5Mappings:
    def test_detects_dense_and_moe_ffn_layer_indices(self):
        model = _make_step3p5_model(num_layers=6, moe_start=2)
        dense_indices, moe_indices = _detect_step3p5_ffn_layer_indices(model)
        assert dense_indices == [0, 1]
        assert moe_indices == [2, 3, 4, 5]

    def test_builds_split_dense_and_moe_mappings(self):
        model = _make_step3p5_model(num_layers=6, moe_start=2)
        mappings = build_step3p5_mappings(model)
        assert mappings is not None
        assert len(mappings) == 5

        dense_mapping = mappings[2]
        assert "0|1" in dense_mapping.smooth_layer
        assert all("mlp." in balance for balance in dense_mapping.balance_layers)

        moe_mapping = mappings[3]
        assert "2|3|4|5" in moe_mapping.smooth_layer
        assert any("moe." in balance for balance in moe_mapping.balance_layers)
        assert any("share_expert." in balance for balance in moe_mapping.balance_layers)
        assert not any("mlp." in balance for balance in moe_mapping.balance_layers)

    def test_dynamic_registry_model_uses_step3p5_dynamic_path(self):
        model = _make_step3p5_model(num_layers=6, moe_start=2)
        model.__class__ = type("Step3p5ForCausalLM", (model.__class__,), {})
        assert model.__class__.__name__ in AWQ_DYNAMIC_MAPPING_REGISTRY

        mappings = get_layer_mappings_from_model(model)
        assert len(mappings) == 5
        assert mappings[2].smooth_layer.endswith("(0|1)\\.post_attention_layernorm$")
        assert mappings[3].smooth_layer.endswith(
            "(2|3|4|5)\\.post_attention_layernorm$"
        )

    def test_step3p5_mappings_resolve_per_layer(self):
        model = _make_step3p5_model(num_layers=6, moe_start=2)
        mappings = build_step3p5_mappings(model)
        assert mappings is not None

        apply_quantization_config(
            model,
            config=QuantizationModifier(
                scheme="W4A16_ASYM"
            ).resolve_quantization_config(),
        )
        awq = AWQModifier(mappings=mappings)
        awq._set_resolved_mappings(model)

        post_attention_mappings = [
            mapping
            for mapping in awq._resolved_mappings
            if mapping.smooth_name.endswith("post_attention_layernorm")
        ]
        assert len(post_attention_mappings) == 6

        dense_mappings = [
            mapping
            for mapping in post_attention_mappings
            if ".layers.0." in mapping.smooth_name
            or ".layers.1." in mapping.smooth_name
        ]
        assert len(dense_mappings) == 2
        assert all(
            all(".mlp." in name for name in mapping.balance_names)
            for mapping in dense_mappings
        )

        moe_mappings = [
            mapping
            for mapping in post_attention_mappings
            if ".layers.2." in mapping.smooth_name
            or ".layers.3." in mapping.smooth_name
            or ".layers.4." in mapping.smooth_name
            or ".layers.5." in mapping.smooth_name
        ]
        assert len(moe_mappings) == 4
        assert all(
            all(
                ".moe." in name or ".share_expert." in name
                for name in mapping.balance_names
            )
            for mapping in moe_mappings
        )


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

    def test_nanbeige_model_uses_static_default_mappings(self):
        model = _make_standard_model()
        model.__class__ = type("NanbeigeForCausalLM", (model.__class__,), {})

        model_name = model.__class__.__name__

        assert AWQ_MAPPING_REGISTRY[model_name] == default_mappings
        assert get_layer_mappings_from_model(model) == default_mappings

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

    def test_whisper_model_uses_static_whisper_mappings(self):
        model = _make_standard_model()
        model.__class__ = type(
            "WhisperForConditionalGeneration", (model.__class__,), {}
        )

        model_name = model.__class__.__name__

        assert AWQ_MAPPING_REGISTRY[model_name] == _whisper_mappings
        assert get_layer_mappings_from_model(model) == _whisper_mappings


def test_whisper_mapping_regex_matches_real_module_tree():
    """WhisperForConditionalGeneration had no AWQ mapping entry, so it silently
    fell back to `default_mappings`, whose balance layers (q_proj/k_proj/v_proj,
    gate_proj/up_proj, down_proj) never match Whisper's actual layer names --
    Whisper's MLP is a plain fc1/fc2 pair (no gate_proj/up_proj/down_proj at
    all), so smoothing was a complete no-op for this arch, not just a naming
    mismatch on part of it.

    Construct a tiny WhisperForConditionalGeneration on the meta device (no HF
    Hub download, no weight allocation) and confirm the newly-registered
    _whisper_mappings regex actually matches its real module names -- this is
    a mechanical port of the already-validated WHISPER_V2_SMOOTHQUANT_MAPPINGS
    (smoothquant/utils.py) into the AWQMapping dataclass shape.
    """
    import re

    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    config = WhisperConfig(
        vocab_size=100,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        num_mel_bins=10,
        max_source_positions=20,
        max_target_positions=20,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        decoder_start_token_id=0,
    )
    with torch.device("meta"):
        model = WhisperForConditionalGeneration(config)

    module_names = [name for name, _ in model.named_modules()]

    for mapping in _whisper_mappings:
        smooth_pat = mapping.smooth_layer.removeprefix("re:")
        smooth_hits = [n for n in module_names if re.search(smooth_pat, n)]
        assert smooth_hits, (
            f"Whisper: smooth pattern {smooth_pat!r} matched no modules; "
            f"sample names: {module_names[:20]}"
        )
        for balance_pat_raw in mapping.balance_layers:
            balance_pat = balance_pat_raw.removeprefix("re:")
            balance_hits = [n for n in module_names if re.search(balance_pat, n)]
            assert balance_hits, (
                f"Whisper: balance pattern {balance_pat!r} matched no "
                f"modules; sample names: {module_names[:20]}"
            )

    # The previous fallback (default_mappings) never fully pairs against
    # Whisper -- its input_layernorm/post_attention_layernorm/gate_proj/
    # up_proj/down_proj patterns all have zero matches here -- but a
    # standalone regex check of default_mappings' pieces in isolation isn't a
    # reliable proxy for "the real resolution fails": default_mappings also
    # has a v_proj->o_proj entry, and v_proj genuinely exists in Whisper too
    # (it's just paired with o_proj, which doesn't -- Whisper's is out_proj).
    # Confirming the real match_modules_set resolution is a no-op is left to
    # the smoothquant precedent's proven evidence, not re-derived by regex
    # here; the positive checks above are this test's real contribution.

    # Regression test for a real bug caught in review: the decoder layer has
    # both self_attn and encoder_attn (cross-attention) blocks with
    # identically-named q/k/v_proj children under the same layer's parent
    # context, so match_modules_set (which groups by shared parent context)
    # would incorrectly pull encoder_attn's projections into the same group
    # as self_attn_layer_norm if the balance patterns weren't scoped to
    # self_attn.*. Call the real resolution function, not a regex
    # approximation of it -- a prior version of this test's regex-only
    # checks couldn't have caught this.
    attn_mapping = _whisper_mappings[0]
    targets = (attn_mapping.smooth_layer, *attn_mapping.balance_layers)
    for group in match_modules_set(model, targets):
        for balance_matches in group[1:]:
            for mod in balance_matches:
                mod_name = next(n for n, m in model.named_modules() if m is mod)
                assert "encoder_attn" not in mod_name, (
                    f"Whisper: self_attn_layer_norm's match_modules_set group "
                    f"incorrectly includes a cross-attention module: "
                    f"{mod_name!r}"
                )
