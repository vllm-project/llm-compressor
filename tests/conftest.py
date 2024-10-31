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

from math import ceil
from typing import Any, Iterable, Optional, Union

import pytest
import torch
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.utils.offload import update_parameter_data


def _get_dim(dim: int, value: torch.Tensor):
    if isinstance(dim, int):
        dim = [dim]
        dim = set(dim)

    reduce_dims = tuple(idx for idx in range(value.ndim) if idx not in dim)
    return reduce_dims


@pytest.fixture
def mock_per_token_calibration():
    def update_scale_zp(module: torch.nn.Module, base_name: str, value: torch.Tensor):
        quantization_scheme = getattr(module, "quantization_scheme", None)
        if not quantization_scheme:
            # no quantization scheme nothing to do
            return

        arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"
        args = getattr(quantization_scheme, arg_name, None)

        dim = _get_dim({0, 1}, value)
        min_val = torch.amin(value, dim=dim, keepdims=True)
        max_val = torch.amax(value, dim=dim, keepdims=True)
        scale, zp = calculate_qparams(min_val, max_val, args)
        scale = scale.reshape((1, 1))
        zp = zp.reshape((1, 1))
        update_parameter_data(module, scale, f"{base_name}_scale")
        update_parameter_data(module, zp, f"{base_name}_zero_point")

    return update_scale_zp


@pytest.fixture
def mock_per_group_calibration():
    def update_scale_zp(
        module: torch.nn.Module, base_name: str, value: torch.Tensor, group_size: int
    ):
        quantization_scheme = getattr(module, "quantization_scheme", None)
        if not quantization_scheme:
            # no quantization scheme nothing to do
            return

        arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"
        args = getattr(quantization_scheme, arg_name, None)

        rows = value.shape[0]
        columns = value.shape[1]
        num_groups = int(ceil(columns / group_size))

        scale = torch.zeros((rows, num_groups), dtype=value.dtype, device=value.device)
        zp_dtype = args.pytorch_dtype()
        zp = torch.zeros((rows, num_groups), dtype=zp_dtype, device=value.device)

        group_sizes = torch.full((num_groups,), group_size, dtype=torch.int)
        end = 0
        for group_index, group_count in enumerate(group_sizes):
            start = end
            end = start + group_count
            dim = _get_dim(
                0,
                value[:, start:end],
            )
            min_val = torch.amin(value, dim=dim, keepdims=True)
            max_val = torch.amax(value, dim=dim, keepdims=True)
            scale_out, zp_out = calculate_qparams(min_val, max_val, args)

            scale[:, group_index] = scale_out.squeeze(1)
            zp[:, group_index] = zp_out.squeeze(1)

        update_parameter_data(module, scale, f"{base_name}_scale")
        update_parameter_data(module, zp, f"{base_name}_zero_point")

    return update_scale_zp


@pytest.fixture
def mock_per_channel_calibration():
    def update_scale_zp(module: torch.nn.Module, base_name: str, value: torch.Tensor):
        quantization_scheme = getattr(module, "quantization_scheme", None)
        if not quantization_scheme:
            # no quantization scheme nothing to do
            return

        arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"

        args = getattr(quantization_scheme, arg_name, None)
        dim = _get_dim(0, value)
        min_val = torch.amin(value, dim=dim, keepdims=True)
        max_val = torch.amax(value, dim=dim, keepdims=True)
        scale, zp = calculate_qparams(min_val, max_val, args)
        update_parameter_data(module, scale, f"{base_name}_scale")
        update_parameter_data(module, zp, f"{base_name}_zero_point")

    return update_scale_zp


@pytest.fixture
def mock_per_tensor_calibration():
    def update_scale_zp(module: torch.nn.Module, base_name: str, value: torch.Tensor):
        quantization_scheme = getattr(module, "quantization_scheme", None)
        if not quantization_scheme:
            # no quantization scheme nothing to do
            return

        arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"
        args = getattr(quantization_scheme, arg_name, None)

        # per tensor quantization just calls calculate_qparams directly
        min_val, max_val = torch.aminmax(value)
        scale, zp = calculate_qparams(min_val, max_val, args)
        update_parameter_data(module, scale, f"{base_name}_scale")
        update_parameter_data(module, zp, f"{base_name}_zero_point")

    return update_scale_zp
