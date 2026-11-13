import torch
from compressed_tensors.compressors.base import compress_module
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
    initialize_module_for_quantization,
    preset_name_to_scheme,
)

from llmcompressor.modifiers.gptq.lutb import GPTQLutBModifier
from llmcompressor.modifiers.gptq.lutb_delayed import (
    quantize_weight_lut_b_delayed,
)


def _lutb_args():
    return QuantizationArgs(
        num_bits=3,
        type=QuantizationType.CODEBOOK,
        strategy=QuantizationStrategy.BLOCK,
        block_structure=[8, 64],
        scale_dtype=torch.uint8,
    )


def test_gptq_lutb_uses_hessian_diagonal_as_importance():
    torch.manual_seed(0)
    module = torch.nn.Linear(64, 16, bias=False)
    importance = torch.linspace(0.1, 4.0, 64)
    hessian = torch.diag(importance)
    args = _lutb_args()

    base = GPTQLutBModifier(use_imatrix=False)
    weighted = GPTQLutBModifier(use_imatrix=True)
    _, base_params, base_fallback = base._quantize_weight(module, args, hessian.clone())
    _, weighted_params, weighted_fallback = weighted._quantize_weight(
        module, args, hessian.clone()
    )

    assert not base_fallback
    assert not weighted_fallback
    assert torch.isfinite(weighted_params["weight"]).all()
    assert not torch.equal(base_params["weight"], weighted_params["weight"])
    assert weighted_params["weight_codebook"].shape == (2, 1, 8)
    assert weighted_params["weight_scale"].shape == (2, 1)


def test_gptq_lutb_codebooks_survive_compression():
    torch.manual_seed(2)
    module = torch.nn.Linear(64, 16, bias=False)
    scheme = preset_name_to_scheme("LUTB", ["Linear"])
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
