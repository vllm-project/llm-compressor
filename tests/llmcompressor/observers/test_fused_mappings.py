import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    initialize_module_for_quantization,
)
from torch.nn import Linear, Module

from llmcompressor.modifiers.quantization.calibration import initialize_observer
from llmcompressor.observers.fused_mappings import (
    FUSED_MODULE_MAPPINGS,
    get_fused_layers,
    resolve_fused_names,
)
from llmcompressor.observers.helpers import fuse_weight_observers

KDA = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj")
MLA = ("q_a_proj", "kv_a_proj_with_mqa")


def test_fused_mappings_well_formed():
    for mapping in FUSED_MODULE_MAPPINGS:
        all_names = mapping.names + mapping.optional_names
        assert len(mapping.names) >= 2, mapping
        assert len(set(all_names)) == len(all_names), mapping


@pytest.mark.parametrize(
    "present,expected",
    [
        # llama attention and mlp
        ({"q_proj", "k_proj", "v_proj", "o_proj"}, [("q_proj", "k_proj", "v_proj")]),
        ({"gate_proj", "up_proj", "down_proj"}, [("gate_proj", "up_proj")]),
        ({"w1", "w2", "w3"}, [("w1", "w3")]),
        # gemma 4 attention_k_eq_v layer: no v_proj
        ({"q_proj", "k_proj", "o_proj"}, [("q_proj", "k_proj")]),
        # kimi delta attention, with and without use_full_rank_gate
        (
            {*KDA, "g_proj", "f_b_proj", "o_proj"},
            [(*KDA, "g_proj")],
        ),
        (
            {*KDA, "g_a_proj", "g_b_proj", "f_b_proj", "o_proj"},
            [KDA],
        ),
        # kimi delta attention as defined in transformers
        (
            {"q_proj", "k_proj", "v_proj", "b_proj", "forget_gate.f_a_proj"},
            [("q_proj", "k_proj", "v_proj", "b_proj", "forget_gate.f_a_proj")],
        ),
        # multi-latent attention, with and without the kimi output gate
        ({*MLA, "q_b_proj", "kv_b_proj", "o_proj", "g_proj"}, [(*MLA, "g_proj")]),
        ({*MLA, "q_b_proj", "kv_b_proj", "o_proj"}, [MLA]),
        # multi-latent attention without q_lora: nothing is packed
        ({"q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "g_proj"}, []),
        # multi-latent attention, mistral checkpoint names
        (
            {"wq_a", "wq_b", "wkv_a_with_mqa", "wkv_b", "wo"},
            [("wq_a", "wkv_a_with_mqa")],
        ),
        # deepseek v4 attention and compressor, transformers and checkpoint names
        ({"q_a_proj", "q_b_proj", "kv_proj", "o_a_proj"}, [("q_a_proj", "kv_proj")]),
        ({"wq_a", "wq_b", "wkv", "wo_a", "wo_b"}, [("wq_a", "wkv")]),
        ({"kv_proj", "gate_proj", "q_b_proj"}, [("kv_proj", "gate_proj")]),
        ({"wkv", "wgate"}, [("wkv", "wgate")]),
        # deepseek sparse attention indexer
        ({"wq_b", "wk", "weights_proj"}, [("wk", "weights_proj")]),
    ],
)
def test_resolve_fused_names(present, expected):
    assert resolve_fused_names(present) == expected


def _quantized_linear():
    linear = Linear(32, 32, bias=False)
    scheme = QuantizationScheme(
        targets=[],
        weights=QuantizationArgs(
            strategy=QuantizationStrategy.TENSOR_GROUP, group_size=16, num_bits=4
        ),
    )
    initialize_module_for_quantization(linear, scheme)
    initialize_observer(linear, base_name="weight")
    return linear


def _module_with(*names):
    module = Module()
    for name in names:
        module.add_module(name, _quantized_linear())
    return module


def _fused_partners(layer: Module) -> set[Module]:
    handler = layer.weight_observer.fusion_handler
    return {partner.module for partner in handler._group}


def test_fuse_weight_observers_kimi_g_proj():
    """
    g_proj is fused with delta attention projections on KDA layers and with
    MLA projections on MLA layers
    """
    kda_names = (*KDA, "g_proj")
    mla_names = (*MLA, "g_proj")
    model = Module()
    model.kda = _module_with(*kda_names, "o_proj")
    model.mla = _module_with(*mla_names, "q_b_proj", "o_proj")

    fuse_weight_observers(model)

    for parent, fused_names in ((model.kda, kda_names), (model.mla, mla_names)):
        for name in fused_names:
            layer = getattr(parent, name)
            expected = {getattr(parent, n) for n in fused_names if n != name}
            assert _fused_partners(layer) == expected, name
        assert not parent.o_proj.weight_observer.fusion_handler.is_fused


def test_fuse_weight_observers_nested_layer():
    """transformers nests Kimi delta attention f_a_proj under forget_gate"""
    model = _module_with("q_proj", "k_proj", "v_proj", "b_proj")
    model.forget_gate = _module_with("f_a_proj", "f_b_proj")

    fuse_weight_observers(model)

    assert _fused_partners(model.forget_gate.f_a_proj) == {
        model.q_proj,
        model.k_proj,
        model.v_proj,
        model.b_proj,
    }
    assert not model.forget_gate.f_b_proj.weight_observer.fusion_handler.is_fused


def test_fuse_weight_observers_requires_same_quantization():
    """vLLM cannot load a packed weight whose layers are quantized differently"""
    model = _module_with(*KDA)
    model.b_proj = Linear(32, 32)

    with pytest.raises(AssertionError, match="no weight observer"):
        fuse_weight_observers(model)


def test_kimi_k3_oneshot_and_model_free_agree():
    """
    The oneshot (module) and model_free_ptq (checkpoint) paths resolve the same
    fused groups on the Kimi-K3 model definition, which mixes delta attention
    layers and MLA layers that both have a g_proj
    """
    from llmcompressor.entrypoints.model_free.microscale import get_fused_names
    from llmcompressor.modeling.kimi_k3 import modeling_kimi_linear
    from llmcompressor.modeling.kimi_k3.configuration_kimi_k3 import KimiLinearConfig
    from llmcompressor.modeling.kimi_k3.modeling_kimi_linear import (
        KimiLinearForCausalLM,
    )

    # the model definition sets its fla imports to None when they fail, e.g.
    # when fla-core is installed without triton
    if modeling_kimi_linear.ShortConvolution is None:
        pytest.skip("Kimi-K3 model definition requires fla-core and triton")

    for use_full_rank_gate in (True, False):
        config = KimiLinearConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=16,
            mla_use_nope=True,
            mla_use_output_gate=True,
            linear_attn_config={
                "kda_layers": [1],
                "full_attn_layers": [2],
                "head_dim": 16,
                "num_heads": 4,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": use_full_rank_gate,
            },
        )
        with torch.device("meta"):
            model = KimiLinearForCausalLM(config)

        module_groups = {
            frozenset(f"{name}.{layer_name}.weight" for layer_name in fused_layers)
            for name, module in model.named_modules()
            for fused_layers in get_fused_layers(module)
        }

        weight_names = [name for name, _ in model.named_parameters()]
        checkpoint_groups = {
            frozenset(group.values()) for group in get_fused_names(weight_names)
        }

        assert module_groups == checkpoint_groups
        kda_gate = "model.layers.0.self_attn.g_proj.weight"
        assert (kda_gate in weight_names) == use_full_rank_gate
        assert (
            frozenset(
                f"model.layers.1.self_attn.{name}.weight" for name in (*MLA, "g_proj")
            )
            in checkpoint_groups
        )
