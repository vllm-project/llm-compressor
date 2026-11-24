from unittest.mock import patch

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.core import create_session
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import (
    AWQMapping,
    AWQModifier,
    QuIPModifier,
    SmoothQuantModifier,
    SpinQuantModifier,
)
from llmcompressor.pipelines import CalibrationPipeline
from llmcompressor.pipelines.sequential.restructure import RestructureHandler

_MODULE_NAME = "model.layers.0.self_attn.q_proj"
_TRANSFORMS = ("AWQ", "SmoothQuant", "QuIP", "SpinQuant")
_QUANTIZERS = ("GPTQ", "imatrix", "static")


def _make_toy_model():
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=16,
        )
    ).eval()
    with torch.no_grad():
        feature_scales = torch.logspace(-1, 1, steps=model.config.hidden_size)
        model.model.embed_tokens.weight.mul_(feature_scales)
    return model


def _make_transform(name):
    if name == "AWQ":
        return AWQModifier(
            mappings=[
                AWQMapping(
                    "model.layers.0.input_layernorm",
                    [_MODULE_NAME],
                )
            ],
            duo_scaling=False,
            n_grid=5,
        )
    if name == "SmoothQuant":
        return SmoothQuantModifier(
            mappings=[([_MODULE_NAME], "model.layers.0.input_layernorm")],
            smoothing_strength=0.5,
        )
    if name == "QuIP":
        return QuIPModifier(
            transform_type="hadamard",
            targets=["re:.*q_proj$"],
            ignore=[],
        )
    if name == "SpinQuant":
        return SpinQuantModifier(rotations=["R1"], transform_type="hadamard")
    raise ValueError(f"Unknown transform: {name}")


def _make_quantizer(name):
    weights = QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy="channel",
        observer="imatrix_mse" if name == "imatrix" else "minmax",
    )
    activations = None
    if name == "GPTQ":
        activations = QuantizationArgs(
            num_bits=8,
            type="int",
            symmetric=True,
            strategy="tensor",
            dynamic=False,
            observer="static_minmax",
        )
    elif name == "static":
        activations = QuantizationArgs(
            num_bits=8,
            type="int",
            symmetric=True,
            strategy="tensor",
            dynamic=False,
            observer="static_minmax",
        )

    scheme = QuantizationScheme(
        targets=["re:.*q_proj$"],
        weights=weights,
        input_activations=activations,
    )
    if name == "GPTQ":
        return GPTQModifier(
            targets=["re:.*q_proj$"],
            config_groups={"group_0": scheme},
        )
    if name in ("imatrix", "static"):
        return QuantizationModifier(config_groups={"group_0": scheme})
    raise ValueError(f"Unknown quantizer: {name}")


def _make_calibration_dataloader():
    samples = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[8, 7, 6, 5, 4, 3, 2, 1]]),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[2, 5, 1, 7, 3, 8, 4, 6]]),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        },
    ]
    return DataLoader(samples, batch_size=None)


def _run_pipeline(transform_name, quantizer_name):
    model = _make_toy_model()
    target_module = model.get_submodule(_MODULE_NAME)
    dataloader = _make_calibration_dataloader()
    modifiers = [] if transform_name is None else [_make_transform(transform_name)]
    modifiers.append(_make_quantizer(quantizer_name))
    statistics = {}

    def capture_gptq_hessians(modifier):
        for module, hessian in modifier._hessians.items():
            module_name = modifier._module_names[module]
            if module_name == _MODULE_NAME:
                statistics["hessian"] = hessian.detach().cpu().clone()
        return original_compress_modules(modifier)

    def capture_quantizer_statistics(modifier, state, event, modules, **kwargs):
        for module in modules.values():
            if module is not target_module:
                continue

            if quantizer_name == "imatrix":
                observer = getattr(module, "weight_observer", None)
                importance = getattr(observer, "_imatrix_sum", None)
                if importance is not None:
                    statistics["imatrix"] = importance.detach().cpu().clone()
            elif quantizer_name == "static":
                observer = getattr(module, "input_observer", None)
                if observer is not None and observer.has_statistics:
                    statistics["static"] = {
                        name: value.detach().cpu().clone()
                        for name, value in observer.get_statistics().items()
                    }

        return original_quantizer_epoch_end(modifier, state, event, modules, **kwargs)

    original_compress_modules = GPTQModifier.compress_modules
    original_quantizer_epoch_end = QuantizationModifier.on_sequential_epoch_end

    with (
        patch.object(GPTQModifier, "compress_modules", capture_gptq_hessians),
        patch.object(
            QuantizationModifier,
            "on_sequential_epoch_end",
            capture_quantizer_statistics,
        ),
        create_session() as session,
    ):
        session.initialize(
            model=model,
            recipe=modifiers,
            calib_data=dataloader,
            start=-1,
            pipeline=None,
            sequential_targets=["LlamaDecoderLayer"],
        )
        pipeline = CalibrationPipeline.from_modifiers(
            session.lifecycle.recipe.modifiers
        )
        pipeline(
            model,
            dataloader,
            DatasetArguments(
                pipeline=None,
                sequential_targets=["LlamaDecoderLayer"],
            ),
            RestructureHandler(model),
        )
        session.finalize()

    return statistics


@pytest.fixture(scope="module")
def plain_quantization_statistics():
    return {name: _run_pipeline(None, name) for name in _QUANTIZERS}


@pytest.mark.unit
@pytest.mark.parametrize("transform_name", _TRANSFORMS)
@pytest.mark.parametrize("quantizer_name", _QUANTIZERS)
def test_transform_precedes_quantization_calibration(
    transform_name, quantizer_name, plain_quantization_statistics
):
    plain = plain_quantization_statistics[quantizer_name]
    transformed = _run_pipeline(transform_name, quantizer_name)

    if quantizer_name == "static":
        assert "static" in plain
        assert "static" in transformed
        assert any(
            not torch.allclose(plain["static"][stat], transformed["static"][stat])
            for stat in ("min_vals", "max_vals")
        ), f"{transform_name} did not change static activation statistics"
    else:
        statistic_name = "hessian" if quantizer_name == "GPTQ" else "imatrix"
        assert statistic_name in plain
        assert statistic_name in transformed
        assert not torch.allclose(
            plain[statistic_name], transformed[statistic_name]
        ), f"{transform_name} did not change {quantizer_name} calibration statistics"
