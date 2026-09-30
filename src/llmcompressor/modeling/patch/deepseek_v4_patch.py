import contextlib
import importlib.util
import os

import torch
import torch.nn.functional as F
from compressed_tensors.compressors.naive_quantized.fp8_block import (
    dequantize_fp8_block_weight,
)
from compressed_tensors.quantization import QuantizationStrategy, QuantizationConfig
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.utils import patch_attr
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention
from transformers.monkey_patching import clear_patch_mapping, register_patch_mapping
from contextlib import contextmanager
from functools import wraps

from compressed_tensors.utils import patch_attr
from transformers import CompressedTensorsConfig

__all__ = [
    "FP8DeepseekV4Attention",
    "patch_dsv4_config",
    "install_dsv4_chat_template",
]


# DeepSeek-V4 does not ship a Jinja `chat_template`; instead it renders conversations
# with a custom `encode_messages()` helper (see the model's `encoding/encoding_dsv4.py`
# and https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731#chat-template). The
# helpers below bridge that encoder to the SWE-smith calibration dataset, which drives
# calibration through `processor.apply_chat_template(...)`.
def _load_dsv4_encoder(model_path: str):
    """Import the `encode_messages` helper bundled with the DSV4 checkpoint."""
    encoder_path = os.path.join(model_path, "encoding", "encoding_dsv4.py")
    spec = importlib.util.spec_from_file_location("encoding_dsv4", encoder_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_swe_smith_messages(messages):
    """Reshape raw SWE-smith trajectory messages for DSV4's `encode_messages`.

    * assistant `thought` -> `reasoning_content` (rendered inside `<think>...</think>`).
      SWE-smith stores identical text in `content` and `thought`, so the duplicated
      `content` is dropped to avoid emitting the deliberation twice.
    * assistant `tool_calls` are already OpenAI-style and pass through unchanged.
    * `tool` observations keep the OpenAI `tool` role (with a scalar `tool_call_id`);
      `encode_messages` merges them into the preceding user turn as `<tool_result>`.
    * list-of-parts content (e.g. tool observations) is flattened to plain text.
    """
    normalized = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            content = "\n\n".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )

        if role == "assistant":
            thought = msg.get("thought") or content or ""
            new_msg = {
                "role": "assistant",
                "reasoning_content": thought,
                "content": "" if content == thought else (content or ""),
            }
            if msg.get("tool_calls"):
                new_msg["tool_calls"] = msg["tool_calls"]
            normalized.append(new_msg)
        elif role == "tool":
            tool_call_ids = msg.get("tool_call_ids") or []
            normalized.append(
                {
                    "role": "tool",
                    "content": content or "",
                    "tool_call_id": (
                        tool_call_ids[0]
                        if tool_call_ids
                        else msg.get("tool_call_id", "")
                    ),
                }
            )
        else:  # system / user
            normalized.append({"role": role, "content": content or ""})

    return normalized


def install_dsv4_chat_template(
    tokenizer, model_path, thinking_mode="thinking", reasoning_effort="low"
):
    """Route `tokenizer.apply_chat_template` through DSV4's `encode_messages`.

    DSV4 has no Jinja chat template, so this installs one backed by the model's
    `encode_messages` helper. This lets the SWE-smith dataset render agent
    trajectories (reasoning + tool calls + tool observations) in the format the
    model was trained on.

    Calibration renders complete trajectories, so `drop_thinking=False` keeps every
    reasoning turn. `encode_messages` inserts the trailing `<｜Assistant｜>` generation
    prompt itself based on message transitions, so `add_generation_prompt` is a no-op.
    """
    encoder = _load_dsv4_encoder(model_path)

    def apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        text = encoder.encode_messages(
            _normalize_swe_smith_messages(conversation),
            thinking_mode=thinking_mode,
            drop_thinking=False,
            reasoning_effort=reasoning_effort,
        )
        if tokenize:
            return tokenizer(text, **kwargs)["input_ids"]
        return text

    tokenizer.apply_chat_template = apply_chat_template
    return tokenizer


@contextmanager
def patch_dsv4_config():
    """
    `moonshotai/Kimi-K3` has an incorrect ignore list. This context patches
    the qconfig on load so that the unquantized modules are properly ignored
    """
    original_init = CompressedTensorsConfig.__init__

    @wraps(original_init)
    def patched_init(self: CompressedTensorsConfig, *args, **kwargs):
        original_init(self, *args, **kwargs)

        config: QuantizationConfig = self.quantization_config
        for name, scheme in config.config_groups.items():
            for target_i, target in list(enumerate(scheme.targets)):
                if "attn" in target:
                    scheme.targets[target_i] = "re:.*attn\\.(q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)$"
                    scheme.targets.append("re:.*attn\\.compressor\\.indexer\\.q_b_proj$")

                if "ffn" in target:
                    scheme.targets[target_i] = "re:.*mlp\\.experts.*"

    with patch_attr(CompressedTensorsConfig, "__init__", patched_init):
        yield



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
