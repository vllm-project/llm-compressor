import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.quant_args import ActivationOrdering

from llmcompressor.modifiers.gptq.gptq_quantize import (
    make_empty_hessian,
    quantize_weight,
)


@pytest.mark.parametrize(
    "actorder",
    [None, ActivationOrdering.WEIGHT, ActivationOrdering.GROUP],
)
@torch.no_grad()
def test_quantize_weight_group_strategy_actorder(actorder):
    module = torch.nn.Linear(8, 6, bias=False)
    quant_args = QuantizationArgs(
        num_bits=4,
        symmetric=True,
        strategy="group",
        group_size=2,
        actorder=actorder,
    )
    module.quantization_scheme = QuantizationScheme(
        targets=["Linear"], weights=quant_args
    )

    hessian = make_empty_hessian(module)
    hessian += torch.diag(
        torch.arange(
            1, hessian.shape[0] + 1, dtype=hessian.dtype, device=hessian.device
        )
    )

    loss, q_param_dict = quantize_weight(
        module=module,
        quant_args=quant_args,
        hessian=hessian,
    )

    assert loss >= 0
    assert q_param_dict["weight"].shape == module.weight.shape
    assert q_param_dict["weight_scale"].shape == (6, 4)
    assert q_param_dict["weight_zero_point"].shape == (6, 4)

    if actorder == ActivationOrdering.GROUP:
        assert q_param_dict["weight_g_idx"].shape == (8,)


@pytest.mark.parametrize(
    "actorder",
    [None, ActivationOrdering.WEIGHT],
)
@torch.no_grad()
def test_quantize_weight_supports_block_strategy(actorder):
    module = torch.nn.Linear(7, 5, bias=False)
    quant_args = QuantizationArgs(
        num_bits=8,
        symmetric=True,
        strategy="block",
        block_structure=[2, 4],
        actorder=actorder,
    )
    module.quantization_scheme = QuantizationScheme(
        targets=["Linear"], weights=quant_args
    )

    hessian = make_empty_hessian(module)
    hessian += torch.eye(hessian.shape[0], dtype=hessian.dtype, device=hessian.device)

    loss, q_param_dict = quantize_weight(
        module=module,
        quant_args=quant_args,
        hessian=hessian,
        blocksize=3,
    )

    assert loss >= 0
    assert q_param_dict["weight"].shape == module.weight.shape
    assert q_param_dict["weight_scale"].shape == (3, 2)
    assert q_param_dict["weight_zero_point"].shape == (3, 2)
    assert "weight_g_idx" not in q_param_dict
