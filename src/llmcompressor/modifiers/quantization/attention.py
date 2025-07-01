from typing import Optional

import torch
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationStatus,
    forward_quantize,
)
from compressed_tensors.transform import TransformBase, TransformLocation
from transformers.modeling_utils import AttentionInterface
from transformers.models.llama.modeling_llama import eager_attention_forward

from llmcompressor.modifiers.quantization.calibration import calibrate_activations


def calibrated_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    for submodule in module.children():
        if isinstance(submodule, TransformBase):
            # 1. apply transforms
            if TransformBase.args.location == TransformLocation.ATTN_Q:
                query = submodule(query)

            if TransformBase.args.location == TransformLocation.ATTN_K:
                key = submodule(key)

            # if TransformBase.args.location == TransformLocation.ATTN_V:
            #     key = submodule(key)

    scheme: Optional[QuantizationScheme] = getattr(module, "quantization_scheme", None)
    status: Optional[QuantizationStatus] = getattr(module, "quantization_status", None)
    if getattr(scheme, "input_activations", None) is not None:
        # 2. calibrate quantization
        if status == QuantizationStatus.CALIBRATION:
            calibrate_activations(module, value=query, base_name="q")
            calibrate_activations(module, value=key, base_name="k")
            calibrate_activations(module, value=value, base_name="v")

        # 3. apply quantization
        if status in (QuantizationStatus.CALIBRATION, QuantizationStatus.FROZEN):
            query = forward_quantize(module, query, "q", scheme.input_activations)
            key = forward_quantize(module, key, "k", scheme.input_activations)
            value = forward_quantize(module, value, "v", scheme.input_activations)

    return eager_attention_forward(
        module, query, key, value, attention_mask, scaling, dropout, **kwargs
    )


AttentionInterface.register("calibrated_attention", calibrated_attention)
