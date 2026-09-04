import math

import pytest
import torch
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationScheme,
)
from loguru import logger

from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.gptq.gptq_quantize import (
    _apply_activation_ordering,
    make_empty_hessian,
    quantize_weight,
)
from llmcompressor.modifiers.quantization.calibration import (
    initialize_observer,
    observe,
)
from tests.testing_utils import requires_compute_capability, requires_gpu


@pytest.mark.parametrize(
    "actorder",
    [None, ActivationOrdering.WEIGHT],
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

    loss, q_param_dict, used_rtn_fallback = _quantize_module(
        module, quant_args, hessian
    )

    assert loss >= 0
    assert not used_rtn_fallback
    assert q_param_dict["weight"].shape == module.weight.shape
    assert q_param_dict["weight_scale"].shape == (6, 4)
    assert q_param_dict["weight_zero_point"].shape == (6, 4)


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

    loss, q_param_dict, used_rtn_fallback = _quantize_module(
        module, quant_args, hessian, blocksize=3
    )

    assert loss >= 0
    assert not used_rtn_fallback
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

    loss, q_param_dict, used_rtn_fallback = _quantize_module(
        module, quant_args, hessian, blocksize=4
    )

    assert loss >= 0
    assert not used_rtn_fallback
    assert q_param_dict["weight"].shape == module.weight.shape
    assert q_param_dict["weight_scale"].shape[0] == module.weight.shape[0]
    assert q_param_dict["weight_zero_point"].shape[0] == module.weight.shape[0]
    assert "weight_g_idx" not in q_param_dict


def _make_channel_quantized_linear(in_features=8, out_features=4):
    module = torch.nn.Linear(in_features, out_features, bias=False)
    quant_args = QuantizationArgs(num_bits=4, symmetric=True, strategy="channel")
    module.quantization_scheme = QuantizationScheme(
        targets=["Linear"], weights=quant_args
    )
    initialize_observer(module, "weight")
    observe(module, "weight")
    return module, quant_args


@torch.no_grad()
def test_quantize_weight_singular_hessian_rtn_fallback():
    module, quant_args = _make_channel_quantized_linear()

    # rank-1 hessian with a nonzero diagonal, so dead-column masking does not
    # repair it; percdamp=0.0 keeps it singular and cholesky fails
    hessian = make_empty_hessian(module) + 1

    loss, q_param_dict, used_rtn_fallback = _quantize_module(
        module, quant_args, hessian, percdamp=0.0
    )

    assert used_rtn_fallback
    assert loss >= 0
    assert q_param_dict["weight"].shape == module.weight.shape


@torch.no_grad()
def test_gptq_rtn_fallback_summary_fires():
    module, quant_args = _make_channel_quantized_linear()

    # qparams written back by compress_module_list must already exist
    module.weight_scale = torch.nn.Parameter(
        torch.empty(4, 1, dtype=module.weight.dtype), requires_grad=False
    )
    module.weight_zero_point = torch.nn.Parameter(
        torch.empty(4, 1, dtype=quant_args.zp_dtype), requires_grad=False
    )

    name = "model.layers.0.self_attn.q_proj"
    modifier = GPTQModifier(dampening_frac=0.0)
    modifier._module_names[module] = name
    modifier._hessians[module] = make_empty_hessian(module) + 1  # singular
    modifier._num_samples[module] = torch.tensor(1.0)

    messages = []
    handler_id = logger.add(messages.append, level="WARNING")
    try:
        modifier.compress_modules()
        modifier._log_rtn_fallback_summary()
    finally:
        logger.remove(handler_id)

    assert modifier._num_compressed_modules == 1
    assert modifier._rtn_fallback_module_names == [name]
    summaries = [str(m) for m in messages if "Hessian inversion failed for" in str(m)]
    assert len(summaries) == 1
    assert "1/1" in summaries[0]
    assert "100.0%" in summaries[0]
    assert "round-to-nearest" in summaries[0]
    assert name in summaries[0]


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
    device = 0 if torch.accelerator.is_available() else "cpu"

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


def _make_observed_linear(in_features, out_features, quant_args, seed=0, device="cpu"):
    torch.manual_seed(seed)
    module = torch.nn.Linear(in_features, out_features, bias=False).to(device)
    module.quantization_scheme = QuantizationScheme(
        targets=["Linear"], weights=quant_args
    )
    initialize_observer(module, "weight")
    observe(module, "weight")
    return module


def _quantization_inputs(modules, hessians):
    qparams = [module.weight_observer.get_qparams() for module in modules]
    global_scales = None
    if qparams[0]["global_scale"] is not None:
        global_scales = torch.stack(
            [qparam["global_scale"].reshape(-1)[0] for qparam in qparams]
        )
    return (
        torch.stack([module.weight for module in modules]),
        torch.stack(hessians),
        torch.stack([qparam["scale"] for qparam in qparams]),
        torch.stack([qparam["zero_point"] for qparam in qparams]),
        global_scales,
    )


def _quantize_module(module, quant_args, hessian, blocksize=128, percdamp=0.01):
    weights, hessians, scales, zero_points, global_scales = _quantization_inputs(
        [module], [hessian]
    )
    weights, hessians, perm = _apply_activation_ordering(
        weights, hessians, quant_args.actorder
    )
    weights, losses, rtn = quantize_weight(
        weights=weights,
        hessians=hessians,
        scale=scales,
        zero_point=zero_points,
        global_scale=global_scales,
        quant_args=quant_args,
        perm=perm,
        blocksize=blocksize,
        percdamp=percdamp,
    )
    q_param_dict = {
        "weight": weights[0],
        "weight_scale": scales[0],
        "weight_zero_point": zero_points[0],
    }
    if global_scales is not None:
        q_param_dict["weight_global_scale"] = global_scales[0]
    return losses[0].item(), q_param_dict, rtn[0].item()


def _make_spd_hessian(in_features, device, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn(in_features, in_features, generator=gen)
    return (mat @ mat.T + torch.eye(in_features)).to(device=device, dtype=torch.float32)


@pytest.mark.parametrize(
    "quant_args",
    [
        QuantizationArgs(num_bits=4, symmetric=True, strategy="group", group_size=16),
        QuantizationArgs(num_bits=4, symmetric=False, strategy="group", group_size=16),
        QuantizationArgs(num_bits=8, symmetric=True, strategy="channel"),
        QuantizationArgs(num_bits=8, symmetric=True, strategy="tensor"),
        QuantizationArgs(num_bits=4, symmetric=True, strategy="tensor"),
        QuantizationArgs(
            num_bits=4, type="float", symmetric=True, strategy="group", group_size=16
        ),
        QuantizationArgs(
            num_bits=4,
            type="float",
            symmetric=True,
            strategy="tensor_group",
            group_size=16,
        ),
    ],
)
@pytest.mark.parametrize("actorder", [None, ActivationOrdering.WEIGHT])
@requires_gpu
@torch.no_grad()
def test_fused_gptq_kernel_matches_eager(quant_args, actorder, monkeypatch):
    """The fused Triton block update must match the eager column loop."""
    if actorder is not None:
        quant_args.actorder = actorder

    hessian = _make_spd_hessian(64, "cuda", seed=1)

    monkeypatch.setenv("LLMCOMPRESSOR_DISABLE_GPTQ_TRITON", "1")
    loss_eager, q_eager, rtn_eager = _quantize_module(
        _make_observed_linear(64, 48, quant_args, seed=0, device="cuda"),
        quant_args,
        hessian.clone(),
    )
    monkeypatch.delenv("LLMCOMPRESSOR_DISABLE_GPTQ_TRITON")
    loss_fused, q_fused, rtn_fused = _quantize_module(
        _make_observed_linear(64, 48, quant_args, seed=0, device="cuda"),
        quant_args,
        hessian.clone(),
    )

    assert rtn_eager == rtn_fused
    assert torch.allclose(q_eager["weight"], q_fused["weight"], rtol=1e-4, atol=1e-5), (
        (q_eager["weight"] - q_fused["weight"]).abs().max()
    )
    assert math.isclose(loss_eager, loss_fused, rel_tol=1e-4, abs_tol=1e-5)


@pytest.mark.parametrize(
    "quant_args",
    [
        QuantizationArgs(num_bits=4, symmetric=True, strategy="group", group_size=16),
        QuantizationArgs(num_bits=4, symmetric=False, strategy="group", group_size=16),
        QuantizationArgs(num_bits=8, symmetric=True, strategy="channel"),
        QuantizationArgs(
            num_bits=4, type="float", symmetric=True, strategy="group", group_size=16
        ),
        QuantizationArgs(
            num_bits=4,
            type="float",
            symmetric=True,
            strategy="tensor_group",
            group_size=16,
        ),
    ],
)
@pytest.mark.parametrize("actorder", [None, ActivationOrdering.WEIGHT])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_quantize_weight_batch_matches_single(quant_args, actorder, device):
    """Batched GPTQ over same-shape modules must match per-module solves."""
    if device == "cuda" and not torch.accelerator.is_available():
        pytest.skip("requires CUDA")
    if actorder is not None:
        quant_args.actorder = actorder

    num_modules = 4
    single_modules = [
        _make_observed_linear(64, 48, quant_args, seed=seed, device=device)
        for seed in range(num_modules)
    ]
    batched_modules = [
        _make_observed_linear(64, 48, quant_args, seed=seed, device=device)
        for seed in range(num_modules)
    ]
    hessians = [
        _make_spd_hessian(64, device, seed=100 + seed) for seed in range(num_modules)
    ]

    single_results = [
        _quantize_module(module, quant_args, h.clone())
        for module, h in zip(single_modules, hessians)
    ]
    weights, batched_hessians, scales, zero_points, global_scales = (
        _quantization_inputs(batched_modules, [h.clone() for h in hessians])
    )
    weights, batched_hessians, perm = _apply_activation_ordering(
        weights, batched_hessians, quant_args.actorder
    )
    batched_weights, batched_losses, batched_rtn = quantize_weight(
        weights=weights,
        hessians=batched_hessians,
        scale=scales,
        zero_point=zero_points,
        global_scale=global_scales,
        quant_args=quant_args,
        perm=perm,
    )

    for idx, single in enumerate(single_results):
        s_loss, s_params, s_rtn = single
        assert s_rtn == batched_rtn[idx].item()
        assert torch.allclose(
            s_params["weight"], batched_weights[idx], rtol=1e-4, atol=1e-5
        ), f"module {idx} weight mismatch"
        assert torch.allclose(
            s_params["weight_scale"], scales[idx]
        ), f"module {idx} scale mismatch"
        assert math.isclose(
            s_loss, batched_losses[idx].item(), rel_tol=1e-4, abs_tol=1e-5
        ), f"module {idx} loss mismatch"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_quantize_weight_batch_rtn_fallback(device):
    """A singular hessian in one batch slice falls back to RTN for that slice
    only."""
    if device == "cuda" and not torch.accelerator.is_available():
        pytest.skip("requires CUDA")
    quant_args = QuantizationArgs(
        num_bits=4, symmetric=True, strategy="group", group_size=16
    )
    modules = [
        _make_observed_linear(64, 48, quant_args, seed=seed, device=device)
        for seed in range(3)
    ]
    hessians = [_make_spd_hessian(64, device, seed=100 + seed) for seed in range(3)]
    hessians[1] = torch.ones(64, 64, device=device)  # singular

    weights, batched_hessians, scales, zero_points, global_scales = (
        _quantization_inputs(modules, hessians)
    )
    with pytest.raises(torch.linalg.LinAlgError):
        quantize_weight(
            weights=weights,
            hessians=batched_hessians,
            scale=scales,
            zero_point=zero_points,
            global_scale=global_scales,
            quant_args=quant_args,
            percdamp=0.0,
        )

    observe(modules, base_name="weight")
    results = [
        _quantize_module(module, quant_args, hessian, percdamp=0.0)
        for module, hessian in zip(modules, hessians)
    ]
    assert [result[2] for result in results] == [False, True, False]


@torch.no_grad()
def test_compress_module_list_batches_same_shape(tmp_path):
    """compress_module_list groups same-shape modules into one batched solve
    and leaves odd ones on the single-matrix path."""
    quant_args = QuantizationArgs(
        num_bits=4, symmetric=True, strategy="group", group_size=16
    )

    def make_module(in_features, out_features, seed):
        module = _make_observed_linear(in_features, out_features, quant_args, seed)
        module.weight_scale = torch.nn.Parameter(
            torch.empty(out_features, in_features // 16), requires_grad=False
        )
        module.weight_zero_point = torch.nn.Parameter(
            torch.empty(out_features, in_features // 16, dtype=quant_args.zp_dtype),
            requires_grad=False,
        )
        return module

    # three same-shape modules (one batch) + one different shape (singleton)
    modules = [make_module(64, 48, seed) for seed in range(3)]
    modules.append(make_module(32, 48, seed=3))

    modifier = GPTQModifier()
    for idx, module in enumerate(modules):
        modifier._module_names[module] = f"model.layers.0.experts.{idx}"
        modifier._hessians[module] = _make_spd_hessian(
            module.weight.shape[1], module.weight.device, seed=idx
        )
        modifier._num_samples[module] = torch.tensor(1.0)

    # force pure eager+single path so CPU runs deterministically cover the
    # batching decision logic, not the kernel
    modifier.batched_quantization = False
    modifier.compress_modules()
    assert modifier._num_compressed_modules == 4

    # re-fill and run with batching enabled
    modifier._num_compressed_modules = 0
    modules_b = [make_module(64, 48, seed) for seed in range(3)]
    modules_b.append(make_module(32, 48, seed=3))
    modifier._module_names = {
        m: f"model.layers.0.experts.{i}" for i, m in enumerate(modules_b)
    }
    modifier._hessians = {
        m: _make_spd_hessian(m.weight.shape[1], m.weight.device, seed=i)
        for i, m in enumerate(modules_b)
    }
    modifier._num_samples = {m: torch.tensor(1.0) for m in modules_b}
    modifier.batched_quantization = True
    modifier.compress_modules()
    assert modifier._num_compressed_modules == 4

    # batched and single results must agree on the shared-seed modules
    for m_single, m_batched in zip(modules, modules_b):
        assert torch.allclose(m_single.weight_scale, m_batched.weight_scale)
        # quantized weights were written back through update_offload_parameter
        assert torch.allclose(m_single.weight, m_batched.weight, rtol=1e-4, atol=1e-5)
