# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps
from math import ceil
from typing import Optional

import torch
from compressed_tensors.quantization.quant_args import (
    DynamicType,
    QuantizationArgs,
    QuantizationStrategy,
    round_to_quantized_type,
)
from compressed_tensors.quantization.quant_config import QuantizationStatus
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.quantization.utils import (
    calculate_range,
    compute_dynamic_scales_and_zp,
)
from torch.nn import Module


__all__ = [
    "quantize",
    "dequantize",
    "fake_quantize",
    "wrap_module_forward_quantized",
    "forward_quantize",
]


@torch.no_grad()
def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
    dtype: Optional[torch.dtype] = None,
    g_idx: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Quantize the input tensor x using the QuantizationStrategy specified in args.
    Quantization can be done per tensor, channel, token or group. For group
    quantization, the group_size must be divisible by the column size. The input scale
    and zero_points are reshaped to support vectorization (Assumes 1 is the
    channel dimension)

    :param x: Input tensor
    :param scale: scale tensor
    :param zero_point: zero point tensor
    :param args: quantization args dictating how to quantize x
    :param dtype: optional dtype to cast the quantized output to
    :param g_idx: optional mapping from column index to group index
    :param global_scale: optional constant to scale the quantization scale during QDQ
    :return: fake quantized tensor
    """

    return _process_quantization(
        x=x,
        scale=scale,
        zero_point=zero_point,
        args=args,
        dtype=dtype,
        do_quantize=True,
        do_dequantize=False,
        g_idx=g_idx,
        global_scale=global_scale,
    )


@torch.no_grad()
def dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    args: Optional[QuantizationArgs] = None,
    dtype: Optional[torch.dtype] = None,
    g_idx: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Dequantize a quantized input tensor x_q based on the strategy specified in args. If
    args is not provided, the strategy will be inferred.

    :param x: quantized input tensor
    :param scale: scale tensor
    :param zero_point: zero point tensor
    :param args: quantization args used to quantize x_q
    :param dtype: optional dtype to cast the dequantized output to
    :param g_idx: optional mapping from column index to group index
    :param global_scale: optional constant to scale the quantization scale during QDQ
    :return: dequantized float tensor
    """
    if args is None:
        if scale.ndim == 0 or scale.ndim == 1:
            args = QuantizationArgs(strategy=QuantizationStrategy.TENSOR)
        elif scale.ndim == 2:
            if scale.shape[1] == 1:
                args = QuantizationArgs(strategy=QuantizationStrategy.CHANNEL)
            # Scale height matches input or is 1 -> group quantization across columns
            #
            # Example 1: scale.shape[0] == 1
            # x_q: (4, 8), scale: (1, 4) -> 2 columns per group
            #
            # Example 2: scale.shape[0] == x_q.shape[0]
            # x_q: (4, 8), scale: (4, 4) -> 2 elements per group (per row)
            elif (scale.shape[0] == 1) or (scale.shape[0] == x_q.shape[0]):
                group_size = int(x_q.shape[1] / scale.shape[1])
                args = QuantizationArgs(
                    strategy=QuantizationStrategy.GROUP, group_size=group_size
                )
            else:
                rows, cols = x_q.shape[-2], x_q.shape[-1]
                block_height = rows // scale.shape[0]  # Rows per block
                block_width = cols // scale.shape[1]  # Columns per block

                args = QuantizationArgs(
                    strategy=QuantizationStrategy.BLOCK,
                    block_structure=[block_height, block_width],
                )
        else:
            raise ValueError(
                f"Could not infer a quantization strategy from scale with {scale.ndim} "
                "dimmensions. Expected 0 or 2 dimmensions."
            )

    if dtype is None:
        dtype = scale.dtype

    return _process_quantization(
        x=x_q,
        scale=scale,
        zero_point=zero_point,
        args=args,
        do_quantize=False,
        do_dequantize=True,
        dtype=dtype,
        g_idx=g_idx,
        global_scale=global_scale,
    )


@torch.no_grad()
def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
    g_idx: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fake quantize the input tensor x by quantizing then dequantizing with
    the QuantizationStrategy specified in args. Quantization can be done per tensor,
    channel, token or group. For group quantization, the group_size must be divisible
    by the column size. The input scale  and zero_points are reshaped to support
    vectorization (Assumes 1 is the channel dimension)

    :param x: Input tensor
    :param scale: scale tensor
    :param zero_point: zero point tensor
    :param args: quantization args dictating how to quantize x
    :param g_idx: optional mapping from column index to group index
    :param global_scale: optional constant to scale the quantization scale during QDQ
    :return: fake quantized tensor
    """
    return _process_quantization(
        x=x,
        scale=scale,
        zero_point=zero_point,
        args=args,
        do_quantize=True,
        do_dequantize=True,
        g_idx=g_idx,
        global_scale=global_scale,
    )


@torch.no_grad()
def _process_quantization(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantizationArgs,
    g_idx: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    do_quantize: bool = True,
    do_dequantize: bool = True,
    global_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q_min, q_max = calculate_range(args, x.device)
    group_size = args.group_size

    # blockwise FP8: quantize per 2D block, supports block_structure for static block
    # quantization
    if args.strategy == QuantizationStrategy.BLOCK:
        original_shape = x.shape
        rows, cols = x.shape[-2], x.shape[-1]
        block_height, block_width = args.block_structure

        # Ensure exact division (tensor dimensions must be divisible by block size)
        if rows % block_height != 0:
            raise ValueError(
                f"Tensor height {rows} is not divisible by block_height {block_height}."
                f" Block quantization requires exact division."
            )
        if cols % block_width != 0:
            raise ValueError(
                f"Tensor width {cols} is not divisible by block_width {block_width}. "
                f"Block quantization requires exact division."
            )

        # reshape into blocks and transpose to make each block contiguous
        num_rows_blocks = rows // block_height
        num_cols_blocks = cols // block_width
        x_blocks = x.reshape(
            num_rows_blocks,
            block_height,
            num_cols_blocks,
            block_width,
        ).transpose(1, 2)

        # expand scale/zero_point for blocks
        sb = scale.unsqueeze(-1).unsqueeze(-1)
        zb = zero_point.unsqueeze(-1).unsqueeze(-1) if zero_point is not None else None
        if do_quantize:
            # quantize blocks
            x_blocks = _quantize(
                x=x_blocks,
                scale=sb,
                zero_point=zb,
                q_min=q_min,
                q_max=q_max,
                args=args,
                dtype=dtype,
                global_scale=global_scale,
            )
        if do_dequantize:
            # dequantize blocks
            x_blocks = _dequantize(
                x_q=x_blocks,
                scale=sb,
                zero_point=zb,
                global_scale=global_scale,
            )
        # restore original shape
        output = x_blocks.transpose(1, 2).reshape(original_shape)
    elif args.strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):

        output_dtype = dtype if dtype is not None else x.dtype
        output = torch.zeros_like(x).to(output_dtype)
        columns = output.shape[-1]

        # TODO: make validation step for inputs

        while scale.ndim < 2:
            # pad scale and zero point dims for slicing
            scale = scale.unsqueeze(1)
            zero_point = zero_point.unsqueeze(1) if zero_point is not None else None

        if columns >= group_size:
            if columns % group_size != 0:
                raise ValueError(
                    "tensor column shape must be divisble "
                    f"by the given group_size {group_size} but got {columns}"
                )

        # support column-order (default) quantization as well as other orderings
        # such as activation ordering. Below checks if g_idx has been initialized
        is_column_order = g_idx is None or -1 in g_idx
        if is_column_order:
            num_groups = int(ceil(columns / group_size))
            group_sizes = torch.full((num_groups,), group_size, dtype=torch.int)

        else:
            group_indices, group_sizes = torch.unique(g_idx, return_counts=True)
            group_sizes = group_sizes[torch.argsort(group_indices)]

            perm = torch.argsort(g_idx)
            x = x.index_select(-1, perm)

        # Maintain all dimensions except the last dim, which is divided by group_size
        reshaped_dims = (
            ceil(x.shape[-1] / group_size),
            group_size,
        )
        x = x.unflatten(-1, reshaped_dims)

        if do_quantize:
            output = _quantize(
                x=x,
                scale=scale.unsqueeze(-1),
                zero_point=zero_point.unsqueeze(-1) if zero_point is not None else None,
                dtype=dtype,
                global_scale=global_scale,
                q_min=q_min,
                q_max=q_max,
                args=args,
            )

        if do_dequantize:
            input = output if do_quantize else x
            output = _dequantize(
                x_q=input,
                scale=scale.unsqueeze(-1),
                zero_point=zero_point.unsqueeze(-1) if zero_point is not None else None,
                global_scale=global_scale,
            )

        output = output.flatten(start_dim=-2)
        output = output.to(output_dtype)

        if not is_column_order:
            inv_perm = torch.argsort(perm)
            output = output.index_select(-1, inv_perm)

    else:  # covers tensor, channel, token, and attn_head strategies
        if do_quantize:
            output = _quantize(
                x=x,
                scale=scale,
                zero_point=zero_point,
                q_min=q_min,
                q_max=q_max,
                args=args,
                dtype=dtype,
                global_scale=global_scale,
            )
        if do_dequantize:
            output = _dequantize(
                output if do_quantize else x,
                scale=scale,
                zero_point=zero_point,
                global_scale=global_scale,
            )

    return output


def wrap_module_forward_quantized(module: Module, scheme: QuantizationScheme):
    # expects a module already initialized and injected with the parameters in
    # initialize_module_for_quantization
    if hasattr(module.forward, "__func__"):
        forward_func_orig = module.forward.__func__
    else:
        forward_func_orig = module.forward.func

    @wraps(forward_func_orig)  # ensures docstring, names, etc are propagated
    def wrapped_forward(self, *args, **kwargs):
        if not getattr(module, "quantization_enabled", True):
            # quantization is disabled on forward passes, return baseline
            # forward call
            return forward_func_orig.__get__(module, module.__class__)(*args, **kwargs)

        input_ = args[0]

        compressed = module.quantization_status == QuantizationStatus.COMPRESSED

        if scheme.input_activations is not None:
            # prehook should calibrate activations before forward call
            input_ = forward_quantize(module, input_, "input", scheme.input_activations)

        if scheme.weights is not None and not compressed:
            # calibrate and (fake) quantize weights when applicable
            unquantized_weight = self.weight.data.clone()
            self.weight.data = forward_quantize(
                module, self.weight, "weight", scheme.weights
            )

        # perform wrapped forward call
        output = forward_func_orig.__get__(module, module.__class__)(
            input_, *args[1:], **kwargs
        )

        # restore back to unquantized_value
        if scheme.weights is not None and not compressed:
            self.weight.data = unquantized_weight

        if scheme.output_activations is not None:
            # forward-hook should calibrate/forward_quantize
            if (
                module.quantization_status == QuantizationStatus.CALIBRATION
                and not scheme.output_activations.dynamic
            ):
                return output

            output = forward_quantize(
                module, output, "output", scheme.output_activations
            )
        return output

    # bind wrapped forward to module class so reference to `self` is correct
    bound_wrapped_forward = wrapped_forward.__get__(module, module.__class__)
    # set forward to wrapped forward
    setattr(module, "forward", bound_wrapped_forward)


def forward_quantize(
    module: Module, value: torch.Tensor, base_name: str, args: "QuantizationArgs"
) -> torch.Tensor:

    # in compressed mode, the weight is already compressed and quantized so we don't
    # need to run fake quantization
    if (
        module.quantization_status == QuantizationStatus.COMPRESSED
        and base_name == "weight"
    ):
        return value

    if value.numel() == 0:
        # if the tensor is empty,
        # skip quantization
        return value

    g_idx = getattr(module, "weight_g_idx", None)
    global_scale = getattr(module, f"{base_name}_global_scale", None)

    if args.dynamic in (True, DynamicType.LOCAL):
        # dynamic quantization - determine the scale/zp on the fly
        scale, zero_point = compute_dynamic_scales_and_zp(
            value=value, args=args, module=module, global_scale=global_scale
        )
    else:
        # static quantization - get scale and zero point from layer
        scale = getattr(module, f"{base_name}_scale")
        zero_point = getattr(module, f"{base_name}_zero_point", None)

    return fake_quantize(
        x=value,
        scale=scale,
        zero_point=zero_point,
        args=args,
        g_idx=g_idx,
        global_scale=global_scale,
    )


@torch.no_grad()
def _quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args: QuantizationArgs,
    dtype: Optional[torch.dtype] = None,
    global_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    # if a global scale is optionally provided, use it
    # to further scale the local `scale` parameter
    if global_scale is not None:
        scale = scale.to(global_scale.dtype) / global_scale

    scaled = x / scale

    if zero_point is not None:
        scaled += zero_point.to(x.dtype)

    # clamp first because cast isn't guaranteed to be saturated (ie for fp8)
    clamped_value = torch.clamp(
        scaled,
        q_min,
        q_max,
    )
    quantized_value = round_to_quantized_type(clamped_value, args)

    if dtype is not None:
        quantized_value = quantized_value.to(dtype)

    return quantized_value


@torch.no_grad()
def _dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor = None,
    dtype: Optional[torch.dtype] = None,
    global_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    # if a global scale is optionally provided, use it
    # to further scale the local `scale` parameter
    if global_scale is not None:
        scale = scale.to(global_scale.dtype) / global_scale

    dequant_value = x_q.to(scale.dtype)

    if zero_point is not None:
        dequant_value = dequant_value - zero_point.to(scale.dtype)

    dequant_value = dequant_value * scale

    if dtype is not None:
        dequant_value = dequant_value.to(dtype)

    return dequant_value
