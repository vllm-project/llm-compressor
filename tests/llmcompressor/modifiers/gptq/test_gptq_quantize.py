import pytest
import torch
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationScheme,
)
from compressed_tensors.utils import patch_attr

import llmcompressor.modifiers.gptq.gptq_quantize as gptq_quantize
from llmcompressor.core import active_session, create_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.gptq.gptq_quantize import (
    make_empty_hessian,
    quantize_weight,
)
from llmcompressor.modifiers.quantization.calibration import (
    initialize_observer,
    observe,
)
from tests.testing_utils import requires_compute_capability


def _make_quantize_inputs(strategy: str = "group", device: str = "cpu"):
    torch.manual_seed(0)
    module = torch.nn.Linear(8, 6, bias=False).to(device)
    quant_args_kwargs = dict(
        num_bits=4,
        symmetric=True,
        strategy=strategy,
    )
    if strategy in ("group", "tensor_group"):
        quant_args_kwargs["group_size"] = 2
    if strategy == "block":
        quant_args_kwargs["num_bits"] = 8
        quant_args_kwargs["block_structure"] = [2, 2]

    quant_args = QuantizationArgs(**quant_args_kwargs)
    module.quantization_scheme = QuantizationScheme(
        targets=["Linear"], weights=quant_args
    )
    initialize_observer(module, "weight")
    observe(module, "weight")

    hessian = make_empty_hessian(module)
    A = torch.randn(hessian.shape[0], hessian.shape[1], device=device)
    hessian += A @ A.t()
    hessian += torch.eye(hessian.shape[0], dtype=hessian.dtype, device=device)
    return module, quant_args, hessian


def _make_group_quantize_inputs(device: str = "cpu"):
    return _make_quantize_inputs("group", device)


@torch.no_grad()
@pytest.mark.parametrize(
    "strategy", ["tensor", "channel", "group", "tensor_group", "block"]
)
def test_quantize_weight_compiled_block_path_matches_eager(monkeypatch, strategy):
    module, quant_args, hessian = _make_quantize_inputs(strategy)

    with create_session() as session:
        session.state.enable_compile = False
        eager_loss, eager_qparams = quantize_weight(
            module=module,
            quant_args=quant_args,
            hessian=hessian.clone(),
            blocksize=4,
        )

        calls = 0

        def compiled_block(*args, **kwargs):
            nonlocal calls
            calls += 1
            return gptq_quantize._quantize_block(*args, **kwargs)

        monkeypatch.setattr(gptq_quantize, "_quantize_block_compiled", compiled_block)

        session.state.enable_compile = True
        compiled_loss, compiled_qparams = quantize_weight(
            module=module,
            quant_args=quant_args,
            hessian=hessian.clone(),
            blocksize=4,
        )

    assert calls > 0
    assert compiled_loss == pytest.approx(eager_loss)
    for key in ("weight", "weight_scale", "weight_zero_point"):
        torch.testing.assert_close(compiled_qparams[key], eager_qparams[key])


@torch.no_grad()
def test_quantize_weight_compile_flag_off_uses_eager(monkeypatch):
    module, quant_args, hessian = _make_group_quantize_inputs()

    def compiled_block(*args, **kwargs):
        raise AssertionError("compiled GPTQ block should not be used")

    monkeypatch.setattr(gptq_quantize, "_quantize_block_compiled", compiled_block)

    with create_session() as session:
        session.state.enable_compile = False
        loss, q_param_dict = quantize_weight(
            module=module,
            quant_args=quant_args,
            hessian=hessian,
            blocksize=4,
        )

    assert loss >= 0
    assert q_param_dict["weight"].shape == module.weight.shape


@requires_compute_capability(8, 0)
@torch.no_grad()
def test_quantize_weight_runs_real_compiled_block_path():
    module, quant_args, hessian = _make_group_quantize_inputs(device="cuda")

    with patch_attr(active_session().state, "enable_compile", True):
        loss, q_param_dict = quantize_weight(
            module=module,
            quant_args=quant_args,
            hessian=hessian,
            blocksize=4,
        )

    assert loss >= 0
    assert q_param_dict["weight"].shape == module.weight.shape


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
    initialize_observer(module, "weight")
    observe(module, "weight")

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
    initialize_observer(module, "weight")
    observe(module, "weight")

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


@torch.no_grad()
def test_quantize_weight_channel_actorder_weight():
    # CHANNEL + actorder=WEIGHT should run end-to-end without producing a g_idx
    # (per-channel quantization has no group structure).
    module = torch.nn.Linear(8, 4, bias=False)
    quant_args = QuantizationArgs(
        num_bits=4,
        symmetric=True,
        strategy="channel",
        actorder=ActivationOrdering.WEIGHT,
    )
    module.quantization_scheme = QuantizationScheme(
        targets=["Linear"], weights=quant_args
    )
    initialize_observer(module, "weight")
    observe(module, "weight")

    hessian = make_empty_hessian(module)
    # non-uniform diagonal so activation ordering produces a non-identity perm
    diag = torch.arange(
        1, hessian.shape[0] + 1, dtype=hessian.dtype, device=hessian.device
    )
    hessian += torch.diag(diag)

    loss, q_param_dict = quantize_weight(
        module=module, quant_args=quant_args, hessian=hessian, blocksize=4
    )

    assert loss >= 0
    assert q_param_dict["weight"].shape == module.weight.shape
    assert q_param_dict["weight_scale"].shape[0] == module.weight.shape[0]
    assert q_param_dict["weight_zero_point"].shape[0] == module.weight.shape[0]
    assert "weight_g_idx" not in q_param_dict


@requires_compute_capability(9, 0)  # Requires H100 or higher
@torch.no_grad()
def test_gptq_nvfp4_saves_fused_global_scale(tmp_path):
    """
    Test that GPTQ with NVFP4 (TENSOR_GROUP) properly saves and fuses global_scale.

    This is a regression test for a bug where global_scale was computed but not
    added to q_param_dict, resulting in corrupted saved models.

    Requires H100+ GPU for NVFP4 support.
    """
    from transformers import AutoModelForCausalLM

    from llmcompressor import oneshot

    model_id = "nm-testing/tinysmokellama-3.2"
    output = tmp_path / "nvfp4_gptq_output"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # NVFP4 GPTQ recipe targeting one layer
    recipe = GPTQModifier(
        scheme="NVFP4",
        targets=["Linear"],
        ignore=["lm_head", "re:model\\.layers\\.(?!0\\.).*"],  # Only layer 0
    )

    # Quantize
    oneshot(
        model=model_id,
        dataset="open_platypus",
        output_dir=output,
        recipe=recipe,
        num_calibration_samples=8,
        splits={"calibration": "train[:8]"},
    )

    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(output, device_map=device)

    # Check layer 0 has global_scale attributes
    layer_0 = model.model.layers[0]

    # Check QKV
    for proj_name in ["q_proj", "k_proj", "v_proj"]:
        proj = getattr(layer_0.self_attn, proj_name)
        assert hasattr(
            proj, "weight_global_scale"
        ), f"{proj_name} missing weight_global_scale"

        gs = proj.weight_global_scale.item()
        assert gs > 0, f"{proj_name} global_scale should be positive, got {gs}"
        assert gs > 1e-10, f"{proj_name} global_scale too small: {gs}"
        assert gs < 1e10, f"{proj_name} global_scale too large: {gs}"

    # Verify QKV global_scales are fused (identical)
    q_gs = layer_0.self_attn.q_proj.weight_global_scale.item()
    k_gs = layer_0.self_attn.k_proj.weight_global_scale.item()
    v_gs = layer_0.self_attn.v_proj.weight_global_scale.item()

    assert abs(q_gs - k_gs) < 1e-6, f"QKV not fused: Q={q_gs}, K={k_gs}"
    assert abs(k_gs - v_gs) < 1e-6, f"QKV not fused: K={k_gs}, V={v_gs}"

    # Check gate/up
    for proj_name in ["gate_proj", "up_proj"]:
        proj = getattr(layer_0.mlp, proj_name)
        assert hasattr(
            proj, "weight_global_scale"
        ), f"{proj_name} missing weight_global_scale"

        gs = proj.weight_global_scale.item()
        assert gs > 0, f"{proj_name} global_scale should be positive, got {gs}"
        assert gs > 1e-10, f"{proj_name} global_scale too small: {gs}"
        assert gs < 1e10, f"{proj_name} global_scale too large: {gs}"

    # Verify gate/up global_scales are fused (identical)
    gate_gs = layer_0.mlp.gate_proj.weight_global_scale.item()
    up_gs = layer_0.mlp.up_proj.weight_global_scale.item()

    assert abs(gate_gs - up_gs) < 1e-6, f"gate/up not fused: gate={gate_gs}, up={up_gs}"

    # Verify QKV and gate/up are NOT fused together
    assert abs(q_gs - gate_gs) > 1e-6, f"QKV and gate/up incorrectly fused: {q_gs}"
