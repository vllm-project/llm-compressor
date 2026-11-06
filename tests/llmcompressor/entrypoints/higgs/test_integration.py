"""
Integration tests for HIGGS mixed-precision quantization.
"""

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    preset_name_to_scheme,
)

import llmcompressor.entrypoints.higgs.base as higgs_base
from llmcompressor.entrypoints.higgs import (
    HiggsMSECollectorConverter,
    compute_fused_layer_mse,
    compute_heuristic_alphas,
    compute_layer_mse,
)
from llmcompressor.entrypoints.higgs.utils import UNQUANTIZED_SCHEME
from llmcompressor.entrypoints.model_free.validate import validate_config


@pytest.fixture
def candidate_schemes():
    """Create sample candidate schemes."""
    return {
        "W4A16": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4, type="int", symmetric=True, strategy="tensor"
            ),
        ),
        "W8A8": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=8, type="int", symmetric=True, strategy="tensor"
            ),
        ),
    }


@pytest.fixture
def sample_tensors():
    """Create sample tensor dict simulating a model shard."""
    torch.manual_seed(42)
    return {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 256),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(128, 256),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(512, 256),
        "model.layers.0.mlp.up_proj.weight": torch.randn(512, 256),
    }


def test_mse_collector_basic(candidate_schemes, sample_tensors, require_higgs_cbc):
    """Test basic MSE collector workflow."""
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
        allow_unquantized=False,
    )

    # Process tensors
    result_tensors = collector.process(sample_tensors)

    # Tensors should be unchanged
    for key in sample_tensors:
        assert torch.allclose(result_tensors[key], sample_tensors[key])

    # MSE matrix should be populated
    assert len(collector.mse_matrix) == 4  # 4 layers
    for layer_mse in collector.mse_matrix.values():
        assert len(layer_mse) == 2  # 2 candidate schemes

    # Create config
    config = collector.create_config()

    # Should have config groups
    assert len(config.config_groups) > 0
    assert config.config_groups is not None
    assert validate_config(config=config, scheme=None, ignore=[]) is config


def test_mse_collector_uses_heuristic(
    candidate_schemes, sample_tensors, require_higgs_cbc
):
    """Test MSE collector's built-in alpha heuristic."""
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
        allow_unquantized=False,
    )

    collector.process(sample_tensors)
    config = collector.create_config()

    # Should generate valid config
    assert config is not None
    assert len(config.config_groups) > 0


def test_mse_collector_validation_rejects_unsupported_tensor_shapes(candidate_schemes):
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
        allow_unquantized=False,
    )

    tensors = {
        "model.layers.0.mlp.down_proj.weight": torch.empty(128, 128, device="meta"),
        "model.layers.0.mamba.conv1d.weight": torch.empty(16, 1, 4, device="meta"),
        "model.layers.1.mamba.conv1d.weight": torch.empty(16, 1, 8, device="meta"),
    }

    with pytest.raises(ValueError) as error:
        collector.validate(tensors)

    message = str(error.value)
    assert "model.layers.0.mamba.conv1d.weight: (16, 1, 4)" in message
    assert "model.layers.1.mamba.conv1d.weight: (16, 1, 8)" in message
    assert "ignore list" in message
    assert "split_fused_moe_experts" in message


def test_mse_collector_validation_splits_known_fused_moe(candidate_schemes):
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
        allow_unquantized=False,
    )
    tensors = {
        "model.layers.0.mlp.gate_up_proj.weight": torch.empty(
            2, 128, 64, device="meta"
        ),
        "model.layers.0.mlp.down_proj.weight": torch.empty(2, 64, 64, device="meta"),
    }

    validated_tensors = collector.validate(tensors)

    assert "model.layers.0.mlp.gate_up_proj.weight" not in validated_tensors
    assert all(tensor.ndim == 2 for tensor in validated_tensors.values())


def test_fused_nvfp4_mse_uses_shared_global_scale():
    scheme = preset_name_to_scheme("NVFP4A16", targets=["Linear"])
    small_weight = torch.linspace(-0.1, 0.1, 256).reshape(16, 16)
    large_weight = torch.linspace(-100, 100, 256).reshape(16, 16)

    independent_mse = compute_layer_mse(small_weight, scheme)
    fused_mse = compute_fused_layer_mse(
        {"q_proj": small_weight, "k_proj": large_weight}, scheme
    )

    # The large fused partner changes the global scale used to quantize q_proj.
    assert fused_mse["q_proj"] != pytest.approx(independent_mse)


def test_mse_collector_allows_unquantized_layers(
    candidate_schemes, sample_tensors, require_higgs_cbc
):
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
        allow_unquantized=True,
    )

    collector.process(sample_tensors)

    assert all(
        layer_mse[UNQUANTIZED_SCHEME] == 0.0
        for layer_mse in collector.mse_matrix.values()
    )

    config = collector.create_config()

    # With no bitwidth constraint, zero-MSE unquantized wins for every layer.
    assert config.config_groups == {}


def test_mse_collector_disallows_unquantized_layers_by_default(
    candidate_schemes, sample_tensors
):
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
    )

    collector.process(sample_tensors)

    assert all(
        UNQUANTIZED_SCHEME not in layer_mse
        for layer_mse in collector.mse_matrix.values()
    )


def test_mse_collector_with_fusion(
    candidate_schemes, sample_tensors, require_higgs_cbc
):
    """Test MSE collector with fusion detection."""
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
        allow_unquantized=False,
    )

    collector.process(sample_tensors)
    config = collector.create_config()

    # Should generate valid config respecting fusion constraints
    assert config is not None
    assert len(config.config_groups) > 0


def test_alpha_heuristic_layer_types():
    """Test alpha heuristic distinguishes layer types."""
    layer_names = [
        "model.layers.0.self_attn.q_proj",  # attention, depth 0
        "model.layers.5.mlp.gate_proj",  # mlp, depth 5
        "model.embed_tokens",  # embedding, depth 0
    ]
    param_counts = {layer: 1000 for layer in layer_names}

    alphas = compute_heuristic_alphas(layer_names, param_counts)

    # All layers should have alphas
    assert len(alphas) == 3

    # Embedding should have highest alpha (1.5 multiplier)
    # Attention should have middle alpha (1.2 multiplier)
    # MLP should have lower alpha (0.9 multiplier)
    # But depth 5 gives depth_factor = 1.25, so mlp might be higher than attn at depth 0

    embed_alpha = alphas["model.embed_tokens"]
    attn_alpha = alphas["model.layers.0.self_attn.q_proj"]
    # Embedding should be highest
    assert embed_alpha > attn_alpha


OLMO_ATTN = "model.layers.0.linear_attn"
OLMO_GROUP = [f"{OLMO_ATTN}.{n}" for n in ("q_proj", "k_proj", "v_proj", "g_proj")]
OLMO_CONTEXT = [f"{OLMO_ATTN}.a_proj", f"{OLMO_ATTN}.b_proj"]


def _decreasing_tensors(names):
    """Weights whose scale decreases by 10x per name, so the first name of a
    fused group sets its global scale"""
    generator = torch.Generator().manual_seed(0)
    return {
        f"{name}.weight": torch.randn(32, 32, generator=generator) * 10**-index
        for index, name in enumerate(names)
    }


@pytest.fixture
def solved_groups(monkeypatch):
    """The fused groups that create_config passes to the ILP"""
    groups = []

    def solve_ilp(mse_matrix, fused_groups, **kwargs):
        groups.extend(fused_groups)
        return {name: "NVFP4A16" for name in mse_matrix}

    monkeypatch.setattr(higgs_base, "solve_ilp_mixed_precision", solve_ilp)
    return groups


def test_mse_collector_sees_fused_context_in_other_shards():
    """
    OLMo-Hybrid q/k/v/g_proj are fused only next to a_proj and b_proj, which
    can be stored in another shard
    """
    collector = HiggsMSECollectorConverter(candidate_schemes=["NVFP4A16"], ignore=[])
    collector.set_weight_names({f"{name}.weight" for name in OLMO_GROUP + OLMO_CONTEXT})

    tensors = _decreasing_tensors(OLMO_GROUP)
    collector.process(tensors)

    g_proj = OLMO_GROUP[-1]
    scheme = collector.candidate_schemes["NVFP4A16"]
    independent_mse = compute_layer_mse(tensors[f"{g_proj}.weight"], scheme)
    # the much larger q_proj sets the global scale shared with g_proj
    assert collector.mse_matrix[g_proj]["NVFP4A16"] != pytest.approx(independent_mse)


def test_mse_collector_sees_ignored_fused_context_in_the_shard(solved_groups):
    """
    Without set_weight_names, ignored layers of the shard are context too, in
    process() and in the ILP groups of create_config()
    """
    collector = HiggsMSECollectorConverter(
        candidate_schemes=["NVFP4A16"], ignore=OLMO_CONTEXT
    )
    tensors = _decreasing_tensors(OLMO_GROUP + OLMO_CONTEXT)
    collector.process(tensors)
    collector.create_config()

    g_proj = OLMO_GROUP[-1]
    scheme = collector.candidate_schemes["NVFP4A16"]
    independent_mse = compute_layer_mse(tensors[f"{g_proj}.weight"], scheme)
    assert collector.mse_matrix[g_proj]["NVFP4A16"] != pytest.approx(independent_mse)
    assert solved_groups == [OLMO_GROUP]


def test_mse_collector_sees_ignored_fused_context(solved_groups):
    """The ILP keeps a fused group whose context layers are ignored"""
    collector = HiggsMSECollectorConverter(
        candidate_schemes=["NVFP4A16"], ignore=OLMO_CONTEXT
    )
    tensors = _decreasing_tensors(OLMO_GROUP + OLMO_CONTEXT)
    collector.set_weight_names(set(tensors))
    collector.process(tensors)
    collector.create_config()

    assert solved_groups == [OLMO_GROUP]


MLA = "model.layers.0.attention"
MLA_LAYERS = [f"{MLA}.{n}" for n in ("q_a_proj", "kv_a_proj_with_mqa", "g_proj")]
KDA = "model.layers.1.attention"
KDA_LAYERS = [f"{KDA}.{n}" for n in ("q_proj", "k_proj", "v_proj", "b_proj")]
KDA_QKV = KDA_LAYERS[:3]


@pytest.mark.parametrize(
    "layers, ignore, expected",
    [
        # an ignored optional layer leaves the required ones fused
        (MLA_LAYERS, [MLA_LAYERS[2]], [MLA_LAYERS[:2]]),
        (MLA_LAYERS, [MLA_LAYERS[0]], []),
        # Bailing MoE V3 delta attention with b_proj ignored and Kimi delta
        # attention with f_a_proj ignored give q/k/v, as on main. vLLM rejects
        # a partly ignored packed NVFP4 layer either way
        (KDA_LAYERS + [f"{KDA}.g_proj", f"{KDA}.f_proj"], [KDA_LAYERS[3]], [KDA_QKV]),
        (KDA_LAYERS + [f"{KDA}.f_a_proj"], [f"{KDA}.f_a_proj"], [KDA_QKV]),
    ],
)
def test_mse_collector_ignored_fused_member(solved_groups, layers, ignore, expected):
    """Ignored layers can decide which mapping applies, but never join a group"""
    collector = HiggsMSECollectorConverter(
        candidate_schemes=["NVFP4A16"], ignore=ignore
    )
    tensors = _decreasing_tensors(layers)
    collector.set_weight_names(set(tensors))
    collector.process(tensors)
    collector.create_config()
    assert solved_groups == expected

    # in each group, the first layer sets the global scale of the others
    scheme = collector.candidate_schemes["NVFP4A16"]
    sharing = {name for group in expected for name in group[1:]}
    for name, mse in collector.mse_matrix.items():
        independent_mse = compute_layer_mse(tensors[f"{name}.weight"], scheme)
        assert (mse["NVFP4A16"] != pytest.approx(independent_mse)) == (
            name in sharing
        ), name
