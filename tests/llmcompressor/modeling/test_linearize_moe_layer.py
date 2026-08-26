import torch
from compressed_tensors.offload import get_cache_init_kwargs, offload_module
from transformers import initialization as init
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modeling.moe.helpers import MoEConfig
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.linearize import (
    get_non_linearized_moes,
    linearize_moe_layer,
)

NUM_TEST_TOKENS = 64
MODULE_MSE = 1e-10


def _make_qwen3_moe_model():
    """Create a simple model with a Qwen3 MoE experts module for testing."""
    config = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
    )

    experts = Qwen3MoeExperts(config)
    init.normal_(experts.gate_up_proj, mean=0.0, std=config.initializer_range)
    init.normal_(experts.down_proj, mean=0.0, std=config.initializer_range)

    model = _DummyMoEModel(experts, config)
    return model, experts, config


class _DummyMoEModel(torch.nn.Module):
    def __init__(self, experts, config):
        super().__init__()
        self.config = config
        self.layer = torch.nn.Module()
        self.layer.mlp = torch.nn.Module()
        self.layer.mlp.experts = experts

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return self.layer.mlp.experts(hidden_states, top_k_index, top_k_weights)


def _make_inputs(moe_config):
    hidden_states = torch.randn(
        NUM_TEST_TOKENS, moe_config.hidden_dim, dtype=moe_config.dtype
    )
    top_k_index = torch.randint(
        0,
        moe_config.num_experts,
        size=(NUM_TEST_TOKENS, moe_config.num_experts_per_tok),
    )
    top_k_weights = torch.randn(
        NUM_TEST_TOKENS, moe_config.num_experts_per_tok, dtype=moe_config.dtype
    )
    return hidden_states, top_k_index, top_k_weights


@torch.no_grad()
def test_linearize_moe_layer_replaces_module():
    """linearize_moe_layer should replace the fused experts module in the model."""
    model, experts, config = _make_qwen3_moe_model()
    moe_lookup = {module: name for name, module in get_non_linearized_moes(model)}

    assert len(moe_lookup) == 1
    subgraph_modules = [experts]

    linearized = linearize_moe_layer(model, subgraph_modules, moe_lookup)
    assert len(linearized) == 1

    new_module = model.layer.mlp.experts
    assert new_module is not experts
    assert isinstance(new_module, LinearExperts2D)


@torch.no_grad()
def test_linearize_moe_layer_output_matches():
    """Linearized layer should produce the same output as the original."""
    model, experts, config = _make_qwen3_moe_model()
    moe_config = MoEConfig.from_config(config)
    inputs = _make_inputs(moe_config)

    true_outputs = experts(*inputs)

    moe_lookup = {module: name for name, module in get_non_linearized_moes(model)}
    linearize_moe_layer(model, [experts], moe_lookup)

    outputs = model(*inputs)
    with moe_calibration_context():
        calib_outputs = model(*inputs)

    assert torch.any(true_outputs != 0), "Bad test setup, output is all zeros"
    assert torch.nn.functional.mse_loss(outputs, true_outputs) < MODULE_MSE
    assert torch.nn.functional.mse_loss(calib_outputs, true_outputs) < MODULE_MSE


@torch.no_grad()
def test_linearize_moe_layer_skips_non_moe():
    """Modules not in moe_lookup should be ignored."""
    model, experts, config = _make_qwen3_moe_model()
    moe_lookup = {module: name for name, module in get_non_linearized_moes(model)}

    non_moe_module = torch.nn.Linear(16, 16)
    linearized = linearize_moe_layer(model, [non_moe_module], moe_lookup)
    assert len(linearized) == 0
    assert model.layer.mlp.experts is experts


@torch.no_grad()
def test_linearize_moe_layer_deferred_offloading():
    """With setup_offloading=False (via linearize_moe_layer), offloading should
    not be set up on the new module. The caller is responsible for that."""
    model, experts, config = _make_qwen3_moe_model()
    moe_lookup = {module: name for name, module in get_non_linearized_moes(model)}

    linearized = linearize_moe_layer(model, [experts], moe_lookup)
    new_module, offload_kwargs = linearized[0]

    for submodule in new_module.modules():
        assert not hasattr(submodule, "_offload_cache"), (
            "Offloading should not be set up when setup_offloading=False"
        )


@torch.no_grad()
def test_linearize_moe_layer_returns_offload_kwargs():
    """linearize_moe_layer should capture offload kwargs from the original module."""
    model, experts, config = _make_qwen3_moe_model()
    original_kwargs = get_cache_init_kwargs(experts)

    moe_lookup = {module: name for name, module in get_non_linearized_moes(model)}
    linearized = linearize_moe_layer(model, [experts], moe_lookup)
    _, returned_kwargs = linearized[0]

    assert returned_kwargs == original_kwargs


@torch.no_grad()
def test_from_experts_module_setup_offloading_false():
    """from_experts_module(setup_offloading=False) should skip offload setup."""
    config = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
    )
    experts = Qwen3MoeExperts(config)
    init.normal_(experts.gate_up_proj, mean=0.0, std=config.initializer_range)
    init.normal_(experts.down_proj, mean=0.0, std=config.initializer_range)

    linear_experts_cls = LinearExperts2D.get_linear_experts_cls(Qwen3MoeExperts)
    linear_moe = linear_experts_cls.from_experts_module(
        experts, config, setup_offloading=False
    )

    for submodule in linear_moe.modules():
        assert not hasattr(submodule, "_offload_cache")

    moe_config = MoEConfig.from_config(config)
    inputs = _make_inputs(moe_config)
    true_outputs = experts(*inputs)
    outputs = linear_moe(*inputs)
    assert torch.nn.functional.mse_loss(outputs, true_outputs) < MODULE_MSE


@torch.no_grad()
def test_view_based_parameter_assignment():
    """Non-transposed experts should use views (shared storage) not copies."""
    config = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
    )
    experts = Qwen3MoeExperts(config)
    init.normal_(experts.gate_up_proj, mean=0.0, std=config.initializer_range)
    init.normal_(experts.down_proj, mean=0.0, std=config.initializer_range)

    assert not experts.is_transposed, "Qwen3MoeExperts should be non-transposed"

    linear_experts_cls = LinearExperts2D.get_linear_experts_cls(Qwen3MoeExperts)
    linear_moe = linear_experts_cls.from_experts_module(
        experts, config, setup_offloading=False
    )

    for i in range(linear_moe.num_experts):
        expert = linear_moe[i]
        assert expert.gate_proj.weight.data_ptr() != 0
        assert expert.up_proj.weight.data_ptr() != 0
        assert expert.down_proj.weight.data_ptr() != 0

        assert expert.gate_proj.weight.untyped_storage().data_ptr() == (
            experts.gate_up_proj.untyped_storage().data_ptr()
        ), f"Expert {i} gate_proj should share storage with original gate_up_proj"

        assert expert.up_proj.weight.untyped_storage().data_ptr() == (
            experts.gate_up_proj.untyped_storage().data_ptr()
        ), f"Expert {i} up_proj should share storage with original gate_up_proj"

        assert expert.down_proj.weight.untyped_storage().data_ptr() == (
            experts.down_proj.untyped_storage().data_ptr()
        ), f"Expert {i} down_proj should share storage with original down_proj"


@torch.no_grad()
def test_linearize_moe_layer_idempotent():
    """Calling linearize_moe_layer twice should be a no-op the second time,
    since the first call removes the module from the moe_lookup."""
    model, experts, config = _make_qwen3_moe_model()
    moe_lookup = {module: name for name, module in get_non_linearized_moes(model)}

    linearized_1 = linearize_moe_layer(model, [experts], moe_lookup)
    assert len(linearized_1) == 1

    new_module = model.layer.mlp.experts
    all_modules = list(model.modules())
    linearized_2 = linearize_moe_layer(model, all_modules, moe_lookup)
    assert len(linearized_2) == 0
    assert model.layer.mlp.experts is new_module


@torch.no_grad()
def test_load_quantizable_moe_fallback_does_not_linearize():
    """The load_quantizable_moe fallback path should not linearize the model.
    Linearization is deferred to the sequential pipeline."""
    from compressed_tensors.utils import patch_attr

    from llmcompressor.modeling.moe import conversion_mappings
    from llmcompressor.modeling.moe.linearize import load_quantizable_moe

    with patch_attr(conversion_mappings, "ARCH_TO_2D_MAPPINGS", []):
        with load_quantizable_moe():
            model, experts, config = _make_qwen3_moe_model()

    non_linearized = get_non_linearized_moes(model)
    assert len(non_linearized) == 1, (
        "Fallback path should leave experts in fused (non-linearized) form"
    )
    assert non_linearized[0][1] is experts
