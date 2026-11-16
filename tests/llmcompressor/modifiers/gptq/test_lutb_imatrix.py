import pytest
import torch
from compressed_tensors.compressors.base import compress_module, decompress_module
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
    enable_quantization,
    initialize_module_for_quantization,
    preset_name_to_scheme,
)
from compressed_tensors.quantization.utils import dequantize_lut_b

from llmcompressor.core import State
from llmcompressor.modifiers.gptq.lutb import GPTQLutBModifier
from llmcompressor.modifiers.gptq.lutb_delayed import (
    quantize_weight_lut_b_delayed,
)
from llmcompressor.modifiers.quantization import QuantizationModifier


def _lutb_args(scale_block_structure=None):
    return QuantizationArgs(
        num_bits=3,
        type=QuantizationType.CODEBOOK,
        strategy=QuantizationStrategy.BLOCK,
        block_structure=[8, 64],
        scale_block_structure=scale_block_structure,
        scale_dtype=torch.uint8,
    )


@pytest.mark.parametrize(
    "preset_name, expected_scale_block",
    [("LUTB", [8, 64]), ("LUTB32", [8, 2])],
)
def test_rtn_modifier_resolves_lutb_presets(preset_name, expected_scale_block):
    model = torch.nn.Sequential(torch.nn.Linear(64, 8, bias=False))
    modifier = QuantizationModifier(targets="Linear", scheme=preset_name)
    modifier.on_initialize(State(model=model))

    scheme = model[0].quantization_scheme
    assert scheme.weights.scale_block_structure == expected_scale_block
    enable_quantization(model[0])
    output = model(torch.randn(2, 64))
    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "scale_block_structure, expected_scale_shape",
    [(None, (2, 1)), ([8, 2], (2, 1, 32))],
)
def test_gptq_lutb_uses_hessian_diagonal_as_importance(
    scale_block_structure, expected_scale_shape, monkeypatch
):
    torch.manual_seed(0)
    module = torch.nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    importance = torch.linspace(0.1, 4.0, 64)
    hessian = torch.diag(importance)
    args = _lutb_args(scale_block_structure)

    from compressed_tensors.quantization.utils import lut_b

    quantize_lut_b = lut_b.quantize_lut_b
    received_importance = []

    def quantize_lut_b_spy(*args, **kwargs):
        received_importance.append(kwargs.get("importance"))
        return quantize_lut_b(*args, **kwargs)

    monkeypatch.setattr(lut_b, "quantize_lut_b", quantize_lut_b_spy)
    base = GPTQLutBModifier(use_imatrix=False)
    weighted = GPTQLutBModifier(use_imatrix=True)
    _, base_params, base_fallback = base._quantize_weight(module, args, hessian.clone())
    _, weighted_params, weighted_fallback = weighted._quantize_weight(
        module, args, hessian.clone()
    )

    assert not base_fallback
    assert not weighted_fallback
    assert torch.isfinite(weighted_params["weight"]).all()
    assert received_importance[0] is None
    torch.testing.assert_close(received_importance[1], importance)
    assert weighted_params["weight_codebook"].shape == (2, 1, 8)
    assert weighted_params["weight_scale"].shape == expected_scale_shape


@pytest.mark.parametrize(
    "preset_name, expected_scale_shape",
    [("LUTB", (2, 1)), ("LUTB32", (2, 1, 32))],
)
def test_gptq_lutb_codebooks_and_scales_survive_compression(
    preset_name, expected_scale_shape
):
    torch.manual_seed(2)
    module = torch.nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    scheme = preset_name_to_scheme(preset_name, ["Linear"])
    initialize_module_for_quantization(module, scheme)
    importance = torch.linspace(0.1, 4.0, 64)
    _, qparams, _ = GPTQLutBModifier(use_imatrix=True)._quantize_weight(
        module, scheme.weights, torch.diag(importance)
    )

    module.weight.data.copy_(qparams["weight"])
    module.register_buffer("weight_codebook", qparams["weight_codebook"])
    module.register_buffer("weight_scale", qparams["weight_scale"])
    compress_module(module)

    torch.testing.assert_close(module.weight_codebook, qparams["weight_codebook"])
    torch.testing.assert_close(module.weight_scale, qparams["weight_scale"])
    packed = module.weight_packed
    expected = dequantize_lut_b(
        packed, module.weight_codebook, scheme.weights, module.weight_scale
    )
    torch.testing.assert_close(expected, qparams["weight"])
    decompress_module(module)
    torch.testing.assert_close(module.weight, expected)


def test_delayed_gptq_lutb_accepts_column_importance():
    torch.manual_seed(1)
    module = torch.nn.Linear(128, 16, bias=False)
    importance = torch.linspace(0.1, 3.0, 128)
    hessian = torch.diag(importance)

    loss, qparams, fallback = quantize_weight_lut_b_delayed(
        module,
        _lutb_args(),
        hessian,
        importance=importance,
    )

    assert loss >= 0
    assert not fallback
    assert qparams["weight"].shape == module.weight.shape
    assert torch.isfinite(qparams["weight"]).all()
    assert qparams["weight_codebook"].shape == (2, 2, 8)
    assert qparams["weight_scale"].shape == (2, 2)


def test_delayed_gptq_lutb_rejects_multiple_scales_per_tile():
    module = torch.nn.Linear(64, 8, bias=False)
    hessian = torch.eye(64)

    with pytest.raises(NotImplementedError, match="one scale per codebook tile"):
        quantize_weight_lut_b_delayed(
            module,
            _lutb_args([8, 2]),
            hessian,
        )
