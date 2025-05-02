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

import shutil
from collections import OrderedDict

import pytest
import torch
from compressed_tensors import FloatQuantizationCompressor
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    apply_quantization_config,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from safetensors.torch import save_file
from torch.nn.modules import Linear, Sequential


def get_dummy_quant_config(strategy, group_size=None):
    config_groups = {
        "group_1": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                strategy=strategy, type="float", group_size=group_size
            ),
        ),
    }
    ignore = ["lm_head"]
    quant_config = QuantizationConfig(
        config_groups=config_groups,
        ignore=ignore,
    )

    return quant_config


def make_dummy_g_idx(columns: int, group_size: int) -> torch.Tensor:
    perm = torch.randperm(columns)
    return torch.tensor([index // group_size for index in range(columns)])[perm]


@pytest.mark.parametrize(
    "strategy,group_size,sc,zp",
    [
        [QuantizationStrategy.TENSOR, None, 0.01, 0],
        [
            QuantizationStrategy.GROUP,
            128,
            torch.rand((512, 8, 1)) * 0.01,
            torch.zeros((512, 8, 1), dtype=torch.int8),
        ],
        [
            QuantizationStrategy.CHANNEL,
            None,
            torch.rand((512, 1)) * 0.01,
            torch.zeros((512, 1), dtype=torch.int8),
        ],
    ],
)
def test_quant_format(strategy, group_size, sc, zp):
    dense_state_dict = {
        "dummy.weight": torch.rand((512, 1024)),
        "dummy.weight_scale": torch.tensor(sc, dtype=torch.float32),
        "dummy.weight_zero_point": torch.tensor(zp, dtype=torch.float32),
    }
    if group_size is not None:
        dense_state_dict["dummy.weight_g_idx"] = make_dummy_g_idx(512, group_size)

    quant_config = get_dummy_quant_config(strategy=strategy, group_size=group_size)

    compressor = FloatQuantizationCompressor(config=quant_config)
    module_name_to_scheme = {"dummy": quant_config.config_groups["group_1"]}
    compressed_state_dict = compressor.compress(
        dense_state_dict, names_to_scheme=module_name_to_scheme
    )

    # state_dict params should be the same, minus the zero_point if symmetric
    assert len(dense_state_dict) == len(compressed_state_dict) + 1

    # check compressed to int8
    assert compressed_state_dict["dummy.weight_scale"].dtype == torch.float32
    assert torch.equal(
        compressed_state_dict["dummy.weight_scale"],
        dense_state_dict["dummy.weight_scale"],
    )
    if group_size is not None:
        assert torch.equal(
            compressed_state_dict["dummy.weight_g_idx"],
            dense_state_dict["dummy.weight_g_idx"],
        )


@pytest.mark.parametrize(
    "strategy,group_size",
    [
        [QuantizationStrategy.TENSOR, None],
        [QuantizationStrategy.CHANNEL, None],
        # Note that group quantization is not supported
    ],
)
def test_reload_match(
    mock_per_group_calibration,
    mock_per_channel_calibration,
    strategy,
    group_size,
    tmp_path,
):
    model = Sequential(
        OrderedDict(
            [
                ("dummy", Linear(512, 1024, bias=None)),
            ]
        )
    )
    quant_config = get_dummy_quant_config(strategy=strategy, group_size=group_size)
    apply_quantization_config(model, quant_config)
    model.dummy.quantization_status = QuantizationStatus.CALIBRATION
    if strategy == QuantizationStrategy.GROUP:
        mock_per_group_calibration(
            model.dummy, base_name="weight", value=model.dummy.weight, group_size=128
        )
    if strategy == QuantizationStrategy.CHANNEL:
        mock_per_channel_calibration(
            model.dummy, base_name="weight", value=model.dummy.weight
        )

    compressor = FloatQuantizationCompressor(config=quant_config)
    module_name_to_scheme = {
        "dummy": quant_config.config_groups["group_1"],
    }
    compressed_state_dict = compressor.compress(
        model.state_dict(), names_to_scheme=module_name_to_scheme
    )
    save_file(compressed_state_dict, tmp_path / "model.safetensors")
    reconstructed_dense_gen = compressor.decompress(
        tmp_path, names_to_scheme=module_name_to_scheme
    )
    reconstructed_dense = {}
    for name, value in reconstructed_dense_gen:
        reconstructed_dense[name] = value

    fake_quant_dummy = fake_quantize(
        model.dummy.weight,
        scale=model.dummy.weight_scale,
        zero_point=model.dummy.weight_zero_point,
        args=module_name_to_scheme["dummy"].weights,
    )
    assert torch.equal(fake_quant_dummy, reconstructed_dense["dummy"].get("weight"))

    shutil.rmtree(tmp_path)
