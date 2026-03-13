import os
import shutil

import pytest
import torch
from accelerate import dispatch_model
from accelerate.accelerator import get_state_dict_offloaded_model
from compressed_tensors import QUANTIZATION_CONFIG_NAME
from compressed_tensors.compressors.format import infer_model_format
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationStatus,
    quantize,
)
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor import oneshot
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)
from llmcompressor.utils import untie_word_embeddings
from tests.testing_utils import requires_gpu


@pytest.mark.parametrize(
    "format,dtype",
    [
        ["dense", torch.bfloat16],
        # NOTE: Int8 Decompression fails for transformers>4.49 due to bug in loading
        # parameters with integers (hf attempts to attach gradients to int params)
        # ["int-quantized", torch.bfloat16],
    ],
)
def test_quant_model_reload(format, dtype, tmp_path):
    recipe_str = (
        "tests/llmcompressor/transformers/compression/recipes/new_quant_simple.yaml"
    )
    model_path = "nm-testing/tinysmokellama-3.2"
    device = "cuda:0" if not torch.cuda.is_available() else "cpu"
    dataset = "open_platypus"
    concatenate_data = False
    num_calibration_samples = 16
    splits = {"calibration": f"train[:{num_calibration_samples}]"}

    # create a quantized model
    model = oneshot(
        model=model_path,
        dataset=dataset,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
        precision=dtype,
        tie_word_embeddings=False,
    )

    # Fetch the oneshot model
    og_state_dict = model.state_dict()
    save_path_compressed = tmp_path / "compressed"

    for name, module in model.named_modules():
        if hasattr(module, "quantization_scheme"):
            assert (
                module.weight.dtype == dtype
            ), f"Module {name} has incorrect weight dtype"
            assert (
                module.quantization_status == QuantizationStatus.FROZEN
            ), f"Module {name} has incorrect quantization status"

    # Save to disk, override format
    model.save_pretrained(
        save_path_compressed,
        quantization_format=format,
        save_compressed=True,
    )

    # Verify config on disk
    config = AutoConfig.from_pretrained(save_path_compressed)
    quant_config = getattr(config, QUANTIZATION_CONFIG_NAME)
    assert quant_config["format"] == format

    decompressed_model = AutoModelForCausalLM.from_pretrained(
        save_path_compressed,
        dtype=dtype,
        quantization_config=CompressedTensorsConfig(run_compressed=False),
    )

    _remove_zp(og_state_dict)  # HACK: remove extra zero points added during quant init
    reconstructed_state_dict = decompressed_model.state_dict()
    assert len(og_state_dict) == len(reconstructed_state_dict)
    for key in og_state_dict.keys():
        dense_tensor = og_state_dict[key].to(device)
        reconstructed_tensor = reconstructed_state_dict[key].to(device)
        assert dense_tensor.dtype == reconstructed_tensor.dtype
        if key.endswith("weight") and format != "dense":
            # we don't expect an exact match for compressed
            diff = torch.abs(dense_tensor - reconstructed_tensor)
            assert not torch.any(diff > 0.01).item()
        else:
            assert torch.equal(dense_tensor, reconstructed_tensor)
    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
    "offload,dtype,tie_word_embeddings,device",
    [
        # dtype
        (False, torch.float16, False, "cpu"),
        (False, torch.float16, True, "cpu"),
        (False, torch.float32, False, "cpu"),
        (False, torch.float32, True, "cpu"),
        # offloading
        (True, torch.float16, False, "cpu"),
        (True, torch.float32, False, "cpu"),
        (True, torch.float16, True, "cpu"),
        (True, torch.float32, True, "cpu"),
    ],
)
def test_model_reload(offload, dtype, tie_word_embeddings, device, tmp_path):
    model_path = "nm-testing/tinysmokellama-3.2"
    save_path = tmp_path / "save_path"

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    if offload:
        model = dispatch_model(model, {"": device}, force_hooks=True)
    else:
        model = model.to(device)

    if not tie_word_embeddings:
        untie_word_embeddings(model)

    modify_save_pretrained(model)
    model.save_pretrained(save_path, safe_serialization=True)

    reloaded = AutoModelForCausalLM.from_pretrained(save_path, dtype="auto")

    model_dict = get_state_dict_offloaded_model(model)
    reloaded_dict = get_state_dict_offloaded_model(reloaded)
    assert model_dict.keys() == reloaded_dict.keys()
    for key in model_dict:
        assert torch.equal(model_dict[key].cpu(), reloaded_dict[key].cpu())


@requires_gpu
@pytest.mark.parametrize(
    "offload,dtype,tie_word_embeddings,device",
    [
        (False, torch.float32, False, "cuda:0"),
        (True, torch.float32, False, "cuda:0"),
        (True, torch.float16, True, "cuda:0"),
        (True, torch.float32, True, "cuda:0"),
    ],
)
def test_model_reload_gpu(offload, dtype, tie_word_embeddings, device, tmp_path):
    test_model_reload(offload, dtype, tie_word_embeddings, device, tmp_path)


class DummyLinearModel(nn.Module):
    """
    A dummy linear model for testing purposes, simulating a quantized linear layer.
    """

    def __init__(self, weights, weight_scale=None, zero_point=None):
        super().__init__()
        out_features, in_features = weights.shape

        # Linear layer without bias
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight = nn.Parameter(weights, requires_grad=True)

        # Attach scale and zero-point if provided
        self.linear.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
        self.linear.weight_zero_point = nn.Parameter(zero_point, requires_grad=False)

    def forward(self, x):
        return self.linear(x)


def _create_quantization_config(
    w_bits=8,
    w_type="int",
    w_strategy="tensor",
    quantize_activations=False,
    a_bits=8,
    a_type="int",
    a_strategy="tensor",
):
    """
    Create a quantization configuration for testing.
    """
    config_dict = {
        "global_compression_ratio": 1.0,
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": w_bits,
                    "strategy": w_strategy,
                    "symmetric": True,
                    "type": w_type,
                },
            }
        },
    }

    if quantize_activations:
        config_dict["config_groups"]["group_0"]["input_activations"] = {
            "num_bits": a_bits,
            "strategy": a_strategy,
            "symmetric": True,
            "type": a_type,
        }

    return QuantizationConfig.model_validate(config_dict)


def _quantization_config_from_string(config_str, q_type):
    """
    Parse quantization config from string and type.
    """
    w_bits = int(config_str[1])
    a_bits = int(config_str[3:])
    quantize_activations = a_bits < 16

    return _create_quantization_config(
        w_bits=w_bits,
        w_type=q_type,
        w_strategy="channel",
        quantize_activations=quantize_activations,
        a_bits=a_bits,
        a_type=q_type,
        a_strategy="tensor",
    )


@pytest.mark.parametrize(
    "quant_style,quant_type,expected_format",
    [
        ("W8A8", "int", "int-quantized"),
        ("W4A16", "int", "pack-quantized"),
        ("W8A16", "int", "pack-quantized"),
        ("W8A8", "float", "float-quantized"),
        ("W8A16", "float", "naive-quantized"),
    ],
)
def test_correct_compressor_inferred(
    quant_style,
    quant_type,
    expected_format,
):
    """Test if the correct compressor is inferred based on quantization"""
    weights = torch.rand(10, 4)

    quantization_config = _quantization_config_from_string(quant_style, quant_type)
    quantization_args = quantization_config.config_groups["group_0"].weights

    scale = (
        torch.ones((weights.shape[0], 1))
        if quantization_args.strategy == "channel"
        else torch.tensor([1.0])
    )
    zero_point = torch.zeros_like(scale)

    quantized_weights = quantize(
        weights, scale=scale, zero_point=zero_point, args=quantization_args
    )

    model = DummyLinearModel(quantized_weights, scale, zero_point)
    model.linear.quantization_scheme = quantization_config.config_groups["group_0"]
    model.linear.quantization_status = QuantizationStatus.FROZEN

    assert infer_model_format(model) == expected_format


def _remove_zp(state_dict: dict) -> dict:
    return {
        key: value
        for key, value in state_dict.items()
        if not key.endswith("zero_point")
    }
