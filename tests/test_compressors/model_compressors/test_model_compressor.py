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

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.config import CompressionFormat, SparsityCompressionConfig
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
)
from safetensors.torch import save_file
from tests.testing_utils import induce_sparsity, requires_hf_quantizer
from transformers import AutoModelForCausalLM


def sparsity_config():
    return {
        "format": "sparse-bitmask",  # dense format is ignored by ModelCompressor
        "global_sparsity": 19.098103233975568,
        "registry_requires_subclass": False,
        "sparsity_structure": "unstructured",
    }


def quantization_config():
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "pack-quantized",
        "global_compression_ratio": 1.891791164021256,
        "ignore": ["lm_head"],
        "quant_method": "compressed-tensors",
        "quantization_status": "frozen",
    }


@pytest.mark.parametrize("s_config", [sparsity_config(), None])
@pytest.mark.parametrize("q_config", [quantization_config(), None])
def test_config_format(s_config, q_config):
    combined_config = _get_combined_config(s_config, q_config)
    assert ModelCompressor.parse_sparsity_config(combined_config) == s_config
    assert ModelCompressor.parse_quantization_config(combined_config) == q_config


@requires_hf_quantizer()
@pytest.mark.parametrize(
    "s_config,q_config",
    [
        (sparsity_config(), quantization_config()),
        (sparsity_config(), None),
        (None, quantization_config()),
        (None, None),
    ],
)
def test_hf_compressor_tensors_config(s_config, q_config, tmp_path):
    from transformers.utils.quantization_config import CompressedTensorsConfig

    combined_config = _get_combined_config(s_config, q_config)
    compression_config = CompressedTensorsConfig(**combined_config)
    compressor = ModelCompressor.from_compression_config(compression_config)

    if s_config is q_config is None:
        assert compressor is None
        return

    s_config = (
        SparsityCompressionConfig.load_from_registry(s_config.get("format"), **s_config)
        if s_config is not None
        else None
    )
    q_config = QuantizationConfig(**q_config) if q_config is not None else None

    s_config_dict = s_config.model_dump() if s_config is not None else None
    q_config_dict = q_config.model_dump() if q_config is not None else None

    assert compressor.sparsity_config == s_config
    assert compressor.quantization_config == q_config

    assert ModelCompressor.parse_sparsity_config(compression_config) == s_config_dict
    assert (
        ModelCompressor.parse_quantization_config(compression_config) == q_config_dict
    )


class DummyLinearModel(nn.Module):
    def __init__(self, weights, weight_scale=None, weight_zero_point=None):
        super(DummyLinearModel, self).__init__()
        out_features, in_features = weights.shape

        # Define a linear layer without bias
        self.linear = nn.Linear(in_features, out_features, bias=False)

        # Set the weights of the linear layer
        self.linear.weight = nn.Parameter(weights, requires_grad=False)

        # Attach weight_scale and weight_zero_point as parameters
        if weight_scale is not None:
            self.linear.weight_scale = nn.Parameter(
                torch.tensor(weight_scale), requires_grad=False
            )
        if weight_zero_point is not None:
            self.linear.weight_zero_point = nn.Parameter(
                torch.tensor(weight_zero_point), requires_grad=False
            )

    def forward(self, x):
        return self.linear(x)


def get_bitmask_sparsity_config(targets=["Linear"]):
    from compressed_tensors import BitmaskConfig

    return BitmaskConfig(
        format="sparse-bitmask",
        global_sparsity=0.7,
        targets=targets,
        sparsity_structure="unstructured",
    )


def create_quantization_config(bits=8, type="int", strategy="tensor"):

    config_dict = {
        "format": "int-quantized",
        "global_compression_ratio": 1.0,
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": bits,
                    "strategy": strategy,
                    "symmetric": True,
                    "type": type,
                },
            }
        },
    }

    return QuantizationConfig.model_validate(config_dict)


@pytest.mark.parametrize("sparsity_config", [get_bitmask_sparsity_config()])
@pytest.mark.parametrize(
    "quantization_config",
    [
        create_quantization_config(bits=8, type="int", strategy="channel"),
        create_quantization_config(bits=8, type="float", strategy="channel"),
    ],
)
def test_composability(tmp_path, sparsity_config, quantization_config):

    model_compressor = ModelCompressor(
        sparsity_config=sparsity_config, quantization_config=quantization_config
    )
    fake_oneshot_model: DummyLinearModel = _get_fake_oneshot_sparse_quantized_model(
        sparsity_config=sparsity_config, quantization_config=quantization_config
    )
    fake_oneshot_model = fake_oneshot_model.to(torch.float32)
    # does both sparse and quantization compression
    compressed_state_dict = model_compressor.compress(fake_oneshot_model)

    save_dir = tmp_path / "model"
    save_dir = _create_dummy_checkpoint(
        compressed_state_dict, save_dir, model_compressor
    )

    decompressed_model = DummyLinearModel(
        torch.zeros_like(fake_oneshot_model.linear.weight)
    )
    decompressed_model = decompressed_model.float()
    model_compressor.decompress(model=decompressed_model, model_path=save_dir)

    # check that the decompressed model is the same as the original model
    _check_state_dicts(fake_oneshot_model.state_dict(), decompressed_model.state_dict())


@pytest.mark.parametrize(
    "sparsity_config, quantization_config, missing, unexpected",
    [
        (
            get_bitmask_sparsity_config(),
            create_quantization_config(bits=8, type="int", strategy="channel"),
            {"linear.weight"},
            {
                "linear.bitmask",
                "linear.compressed",
                "linear.row_offsets",
                "linear.shape",
                "linear.weight_scale",
            },
        )
    ],
)
def test_missing_and_unexpected_keys_on_compression(
    tmp_path, sparsity_config, quantization_config, missing, unexpected
):

    model_compressor = ModelCompressor(
        sparsity_config=sparsity_config, quantization_config=quantization_config
    )
    fake_oneshot_model: DummyLinearModel = _get_fake_oneshot_sparse_quantized_model(
        sparsity_config=sparsity_config, quantization_config=quantization_config
    )

    og_state_dict_keys = set(
        DummyLinearModel(weights=torch.randn(10, 5)).state_dict().keys()
    )
    compressed_state_dict_keys = set(
        model_compressor.compress(fake_oneshot_model).keys()
    )

    assert og_state_dict_keys - compressed_state_dict_keys == missing
    assert compressed_state_dict_keys - og_state_dict_keys == unexpected


class TwoLayerModel(nn.Module):
    def __init__(self):
        super(TwoLayerModel, self).__init__()
        self.layer1 = nn.Linear(10, 10, bias=False)
        self.layer2 = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


@pytest.mark.parametrize(
    "model, sparsity_config, quantization_config, expected",
    [
        (
            TwoLayerModel(),
            get_bitmask_sparsity_config(targets=["re:.*layer1$"]),
            create_quantization_config(bits=8, type="int", strategy="channel"),
            {"layer1.weight"},
        )
    ],
)
def test_get_missing_keys(model, sparsity_config, quantization_config, expected):
    model_compressor = ModelCompressor(
        sparsity_config=sparsity_config, quantization_config=quantization_config
    )

    actual = model_compressor.get_missing_module_keys(model)
    assert len(actual) == len(expected) and all(key in actual for key in expected)


@pytest.mark.parametrize(
    "model, sparsity_config, quantization_config, expected",
    [
        (
            TwoLayerModel(),
            get_bitmask_sparsity_config(targets=["re:.*layer1$"]),
            create_quantization_config(bits=8, type="int", strategy="channel"),
            {
                f"{layer}.{suffix}"
                for layer, suffixes in {
                    "layer1": [
                        "shape",
                        "row_offsets",
                        "weight_zero_point",
                        "weight_g_idx",
                        "bitmask",
                        "weight_scale",
                        "compressed",
                    ],
                    "layer2": ["weight_scale", "weight_zero_point", "weight_g_idx"],
                }.items()
                for suffix in suffixes
            },
        )
    ],
)
def test_get_unexpected_keys(model, sparsity_config, quantization_config, expected):
    model_compressor = ModelCompressor(
        sparsity_config=sparsity_config, quantization_config=quantization_config
    )

    actual = model_compressor.get_unexpected_file_keys(model)
    assert len(actual) == len(expected) and all(key in actual for key in expected)


def _create_dummy_checkpoint(state_dict, save_dir, model_compressor):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, save_dir / "model.safetensors")

    config_file_path = save_dir / "config.json"
    with open(config_file_path, "w") as config_file:
        json.dump({}, config_file, indent=2, sort_keys=True)

    model_compressor.update_config(save_dir)
    return save_dir


def _check_state_dicts(state_dict1, state_dict2):
    for key in state_dict1.keys():
        assert key in state_dict2, f"Missing tensor: {key}"
        if key.endswith("weight"):
            original_tensor = state_dict1[key]
            decompressed_tensor = state_dict2[key].to(original_tensor.dtype)
            diff = torch.abs(original_tensor - decompressed_tensor)
            assert not torch.any(diff > 0.01), f"Max diff: {torch.max(diff)}"


def _get_fake_oneshot_sparse_quantized_model(quantization_config, sparsity_config):
    from compressed_tensors.quantization.lifecycle.forward import quantize

    weights = torch.rand(10, 5)
    sparse_weights = induce_sparsity(weights, sparsity_config.global_sparsity)

    quantization_args = quantization_config.config_groups["group_0"].weights

    if quantization_args.strategy == "channel":
        scale = torch.ones((weights.shape[0], 1))
    elif quantization_args.strategy == "tensor":
        scale = torch.tensor([1.0])

    zero_point = torch.zeros_like(scale)

    quantized_weights = quantize(
        sparse_weights,
        scale=scale,
        zero_point=zero_point,
        args=quantization_args,
    )

    fake_oneshot_model = DummyLinearModel(quantized_weights, scale, zero_point)
    fake_oneshot_model.linear.quantization_scheme = quantization_config.config_groups[
        "group_0"
    ]
    return fake_oneshot_model


def _get_combined_config(s_config, q_config):
    combined = {}

    if q_config is not None:
        combined = deepcopy(q_config)

    if s_config is not None:
        combined["sparsity_config"] = s_config

    return combined


@pytest.mark.parametrize(
    "model_stub,q_format,s_config",
    [
        (
            "nm-testing/llama2.c-stories42M-gsm8k-quantized-only-uncompressed",
            "float-quantized",
            None,
        ),
        (
            "nm-testing/llama2.c-stories42M-gsm8k-sparse-only-uncompressed",
            None,
            "sparse-24-bitmask",
        ),
        (
            "nm-testing/llama2.c-stories42M-gsm8k-stacked-uncompressed",
            "float-quantized",
            "sparse-24-bitmask",
        ),
        (
            "nm-testing/llama2.c-stories15M-ultrachat-mixed-uncompressed",
            "pack-quantized",
            None,
        ),
    ],
)
def test_compress_model(model_stub, q_format, s_config, tmpdir):
    model = AutoModelForCausalLM.from_pretrained(model_stub, torch_dtype=torch.float32)
    compressor = ModelCompressor.from_pretrained_model(model, s_config, [q_format])

    # compress model by eagerly compressing state dict
    true_compressed = dict(compressor.compress(model))
    true_compressed = {key: value.clone() for key, value in true_compressed.items()}

    # compress model directly
    compressor.compress_model(model)
    compressed = dict(model.state_dict())

    # equivalent to eagerly compressing state dict
    assert compressed.keys() == true_compressed.keys()
    for key in compressed.keys():
        assert compressed[key].dtype == true_compressed[key].dtype
        assert torch.all(compressed[key] == true_compressed[key]), f"{key}"


@pytest.mark.parametrize(
    "model_stub,q_format,s_config",
    [
        (
            "nm-testing/llama2.c-stories42M-gsm8k-quantized-only-uncompressed",
            "float-quantized",
            None,
        ),
        (
            "nm-testing/llama2.c-stories42M-gsm8k-sparse-only-uncompressed",
            None,
            "sparse-24-bitmask",
        ),
        (
            "nm-testing/llama2.c-stories42M-gsm8k-stacked-uncompressed",
            "float-quantized",
            "sparse-24-bitmask",
        ),
        (
            "nm-testing/llama2.c-stories15M-ultrachat-mixed-uncompressed",
            "pack-quantized",
            None,
        ),
    ],
)
def test_compress_model_meta(model_stub, q_format, s_config):
    # Load model on CPU to get expected compressed state_dict
    cpu_model = AutoModelForCausalLM.from_pretrained(
        model_stub, torch_dtype=torch.float32
    )
    reference_compressor = ModelCompressor.from_pretrained_model(
        cpu_model, s_config, [q_format]
    )
    # Only stores dtype because meta model does not store values
    expected = {k: v.dtype for k, v in reference_compressor.compress(cpu_model).items()}

    # Load model on meta device
    meta_model = AutoModelForCausalLM.from_pretrained(
        model_stub,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    for module in meta_model.modules():
        if hasattr(module, "to_empty"):
            module.to_empty(device="meta")

    # Compress in-place on meta model
    compressor = ModelCompressor.from_pretrained_model(meta_model, s_config, [q_format])
    compressor.compress_model(meta_model)

    # Compare keys and dtypes
    compressed = dict(meta_model.state_dict())
    assert set(compressed.keys()) == set(expected.keys())
    for key, dtype in expected.items():
        assert compressed[key].dtype == dtype, f"{key} has incorrect dtype"


def test_multiple_quant_compressors():
    model = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Linear(2, 3))
    input_activations = QuantizationArgs(num_bits=8, type="float")
    weights = QuantizationArgs(num_bits=8, type="float")

    scheme_fp8 = QuantizationScheme(
        targets=["Linear"],
        weights=weights,
        input_activations=input_activations,
        format=CompressionFormat.float_quantized.value,
    )

    input_activations = QuantizationArgs(num_bits=4, type="float")
    weights = QuantizationArgs(num_bits=4, type="float")

    scheme_nvfp4 = QuantizationScheme(
        targets=["Linear"],
        weights=weights,
        input_activations=input_activations,
        format=CompressionFormat.nvfp4_pack_quantized.value,
    )

    model[0].quantization_scheme = scheme_fp8
    model[0].quantization_status = "frozen"
    model[1].quantization_scheme = scheme_nvfp4
    model[1].quantization_status = "frozen"

    formats = [scheme_fp8.format, scheme_nvfp4.format]

    compressor = ModelCompressor.from_pretrained_model(model, None, formats)
    assert isinstance(compressor.quantization_compressor, dict)
    assert (
        compressor.quantization_config.format == CompressionFormat.mixed_precision.value
    )
    assert all(format in compressor.quantization_compressor for format in formats)


@pytest.mark.parametrize(
    "model_stub,comp_stub",
    [
        (
            "nm-testing/llama2.c-stories42M-gsm8k-quantized-only-uncompressed",
            "nm-testing/llama2.c-stories42M-gsm8k-quantized-only-compressed",
        ),
        (
            "nm-testing/llama2.c-stories42M-gsm8k-sparse-only-uncompressed",
            "nm-testing/llama2.c-stories42M-gsm8k-sparse-only-compressed",
        ),
        (
            "nm-testing/llama2.c-stories42M-gsm8k-stacked-uncompressed",
            "nm-testing/llama2.c-stories42M-gsm8k-stacked-compressed",
        ),
        (
            "nm-testing/llama2.c-stories15M-ultrachat-mixed-uncompressed",
            "nm-testing/llama2.c-stories15M-ultrachat-mixed-compressed",
        ),
    ],
)
def test_decompress_model(model_stub, comp_stub):
    from transformers.utils.quantization_config import CompressedTensorsConfig

    # decompress from disk
    # NOTE: transformers adds extra zero points if run_compressed=False or w/ sparsity
    # https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/quantizer_compressed_tensors.py#L131-L133
    # however, decompression does not add zero points in non-asymmetric cases
    # in order to normalize for this effect in this test, we remove empty weight zps
    true_decompressed_model = AutoModelForCausalLM.from_pretrained(
        comp_stub,
        quantization_config=CompressedTensorsConfig(run_compressed=False),
        torch_dtype=torch.float32,
    )
    true_decompressed = dict(true_decompressed_model.state_dict())
    true_decompressed = remove_empty_weight_zero_points(true_decompressed)  # see above

    # decompress from memory
    # NOTE there is no other way to load a compressed model into memory, since
    # there is no way to turn off decompression for sparse models
    # https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/quantizer_compressed_tensors.py#L133
    model = AutoModelForCausalLM.from_pretrained(model_stub, torch_dtype=torch.float32)
    compressor = ModelCompressor.from_pretrained(comp_stub)
    compressor.compress_model(model)
    compressor.decompress_model(model)
    decompressed = dict(model.state_dict())

    # remove keys not in model definition
    # NOTE it would be better if compressors only returned keys to keep, rather than
    # relying on the model structure + missing keys to catch and remove them later
    model_keys = true_decompressed_model.state_dict().keys()
    decompressed = {key: val for key, val in decompressed.items() if key in model_keys}

    # equivalent to decompressing from disk
    assert decompressed.keys() == true_decompressed.keys()
    for key in decompressed.keys():
        assert decompressed[key].dtype == true_decompressed[key].dtype
        assert torch.all(decompressed[key] == true_decompressed[key]), f"{key}"


def remove_empty_weight_zero_points(state_dict):
    return {
        name: value
        for name, value in state_dict.items()
        if not (name.endswith("weight_zero_point") and torch.all(value == 0))
    }
