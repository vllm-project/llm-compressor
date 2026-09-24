import contextlib

import torch
import torch.nn.functional as F
from compressed_tensors.compressors.naive_quantized.fp8_block import (
    dequantize_fp8_block_weight,
)
from compressed_tensors.quantization import QuantizationStrategy
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.utils import patch_attr
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention
from transformers.monkey_patching import clear_patch_mapping, register_patch_mapping

__all__ = [
    "FP8DeepseekV4Attention",
    "patch_dsv4_fp8_attention",
]


def _is_fp8_block_compressed(module: torch.nn.Module) -> bool:
    """
    Whether ``module`` is a BLOCK-strategy FP8 linear that is still stored in its
    compressed form (fp8 ``weight`` + per-block ``weight_scale``).
    """
    scheme = getattr(module, "quantization_scheme", None)
    if scheme is None or scheme.weights is None:
        return False

    weight = getattr(module, "weight", None)
    return (
        scheme.weights.strategy == QuantizationStrategy.BLOCK
        and getattr(module, "weight_scale", None) is not None
        and isinstance(weight, torch.Tensor)
        and weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    )


def _fp8_block_eager_forward(
    module: torch.nn.Module, input: torch.Tensor
) -> torch.Tensor:
    """
    Eager block-FP8 linear forward: fake-quantize the activations, dequantize the
    weight block-wise, then run the module's native matmul in a higher precision.

    This mirrors ``FloatQuantizationCompressor``'s eager compressed forward but
    bypasses the Triton fp8 tensor-core / emulation kernels that
    ``ImplBackend`` would otherwise dispatch to. Those kernels fault with an
    illegal memory access on some accelerators (e.g. B200 / sm_100); the fault is
    reported asynchronously at the next CUDA op (e.g. RoPE's ``rotate_half``),
    which is misleading. Running the dequant + matmul in torch keeps calibration
    on the FP8 attention weights numerically faithful and kernel-independent.

    Handles both plain ``nn.Linear`` projections and ``DeepseekV4GroupedLinear``
    (``o_a_proj``), which reshapes the dense weight into per-group blocks and runs
    a batched matmul.
    """
    scheme = module.quantization_scheme

    if scheme.input_activations is not None:
        input = forward_quantize(module, input, "input", scheme.input_activations)

    weight = dequantize_fp8_block_weight(
        module.weight,
        module.weight_scale,
        tuple(scheme.weights.block_structure),
        input.dtype,
    )

    # DeepseekV4GroupedLinear (o_a_proj): per-group batched matmul over a 2D weight
    n_groups = getattr(module, "n_groups", None)
    if n_groups is not None:
        input_shape = input.shape[:-2]
        hidden_dim = input.shape[-1]
        w = weight.view(n_groups, -1, hidden_dim).transpose(1, 2)
        x = input.reshape(-1, n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, n_groups, -1)

    return F.linear(input, weight, getattr(module, "bias", None))


class FP8DeepseekV4Attention(DeepseekV4Attention):
    """
    Drop-in replacement for :class:`DeepseekV4Attention` that runs its BLOCK-FP8
    projections through :func:`_fp8_block_eager_forward` for the duration of the
    attention forward.

    This lets the attention weights stay compressed in FP8 during calibration
    (rather than being decompressed to bf16) without hitting the faulting fp8
    Triton kernels. Registered as a class replacement via
    :func:`patch_dsv4_fp8_attention` so it is installed automatically at load.
    """

    def forward(self, *args, **kwargs):
        with contextlib.ExitStack() as stack:
            # covers q_a_proj / q_b_proj / kv_proj / o_a_proj / o_b_proj and the
            # compressor's indexer.q_b_proj -- any FP8 block linear under this block
            for submodule in self.modules():
                if _is_fp8_block_compressed(submodule):
                    stack.enter_context(
                        patch_attr(
                            submodule,
                            "forward",
                            _fp8_block_eager_forward.__get__(submodule),
                        )
                    )
            return super().forward(*args, **kwargs)


@contextlib.contextmanager
def patch_dsv4_fp8_attention():
    """
    Register :class:`FP8DeepseekV4Attention` so DeepSeek-V4 attention blocks are
    constructed with an FP8-aware forward when the model is loaded.

    Wrap the model load in this context (alongside ``load_context``) to keep the
    attention layers in FP8 through calibration::

        with load_context(), patch_dsv4_fp8_attention():
            model = AutoModelForCausalLM.from_pretrained(model_id, ...)
    """
    register_patch_mapping(
        {DeepseekV4Attention.__name__: FP8DeepseekV4Attention}, overwrite=True
    )
    try:
        yield
    finally:
        clear_patch_mapping()
