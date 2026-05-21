"""
Tests that compression algorithms work correctly with disk-offloaded models.
"""

import pytest
import torch
from compressed_tensors.offload import get_device_map, load_offloaded_model
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.utils import getattr_chain
from datasets import Dataset
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

MODEL_ID = "nm-testing/tinysmokellama-3.2"
NUM_CALIBRATION_SAMPLES = 4
MAX_SEQ_LENGTH = 128
MAX_CPU_MEMORY = 6e5  # 600KB forces ~half the modules to disk on tinysmokellama-3.2


@pytest.fixture
def offloaded_model(tmp_path):
    offload_folder = tmp_path / "offload"
    with load_offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            device_map="auto_offload",
            max_memory={"cpu": MAX_CPU_MEMORY},
            offload_folder=str(offload_folder),
        )

    device_map = get_device_map(model)
    disk_modules = [n for n, (_, off) in device_map.items() if off == "disk"]
    assert len(disk_modules) > 0, "No modules were disk-offloaded"

    yield model
    del model


@pytest.fixture
def dataset():
    return Dataset.from_dict({"text": ["Paris is the capital of France. " * 16] * 16})


def _get_weight_snapshot(model):
    return {
        name: param.detach().clone().float().cpu()
        for name, param in model.named_parameters()
        if "weight" in name and param.numel() > 0
    }


def _assert_weights_changed(before, after, targets):
    """Assert that at least one weight matching any target pattern changed."""
    changed = []
    for name in before:
        if any(t in name for t in targets):
            if not torch.equal(before[name], after[name]):
                changed.append(name)
    assert len(changed) > 0, (
        f"No weights matching {targets} were modified. "
        "update_offload_parameter may not be persisting changes to disk."
    )


@pytest.mark.smoke
@pytest.mark.integration
class TestDiskOffloadQuantization:
    def test_quantization_modifier(self, offloaded_model, dataset):
        """QuantizationModifier computes weight_scale and weight_zero_point
        via update_offload_parameter during calibration."""
        oneshot(
            model=offloaded_model,
            dataset=dataset,
            recipe=QuantizationModifier(
                targets="Linear",
                scheme="W8A16",
                ignore=["lm_head"],
            ),
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
        )

        q_proj = offloaded_model.model.layers[0].self_attn.q_proj
        assert hasattr(q_proj, "weight_scale"), "weight_scale not found on q_proj"
        assert hasattr(q_proj, "quantization_scheme")

        # Verify a disk-offloaded layer also got quantized
        q_proj_3 = offloaded_model.model.layers[3].self_attn.q_proj
        assert hasattr(
            q_proj_3, "weight_scale"
        ), "weight_scale not found on disk-offloaded layer"

    def test_gptq_modifier(self, offloaded_model, dataset):
        """GPTQModifier updates weight, weight_scale, weight_zero_point
        via update_offload_parameter after hessian-based quantization."""
        before = _get_weight_snapshot(offloaded_model)

        recipe = GPTQModifier(
            ignore=["lm_head"],
            config_groups={
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=4, strategy="channel", symmetric=True
                    ),
                ),
            },
        )

        oneshot(
            model=offloaded_model,
            dataset=dataset,
            recipe=recipe,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
        )

        after = _get_weight_snapshot(offloaded_model)
        _assert_weights_changed(before, after, ["q_proj", "k_proj", "v_proj"])

        q_proj = offloaded_model.model.layers[0].self_attn.q_proj
        assert hasattr(q_proj, "quantization_scheme")


@pytest.mark.smoke
@pytest.mark.integration
class TestDiskOffloadTransforms:
    def test_smoothquant_modifier(self, offloaded_model, dataset):
        """SmoothQuantModifier updates weight and bias on both smooth_layer and
        balance_layers via update_offload_parameter."""
        before = _get_weight_snapshot(offloaded_model)

        oneshot(
            model=offloaded_model,
            dataset=dataset,
            recipe=SmoothQuantModifier(smoothing_strength=0.5),
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
        )

        after = _get_weight_snapshot(offloaded_model)
        _assert_weights_changed(before, after, ["input_layernorm", "q_proj"])

    def test_awq_modifier(self, offloaded_model, dataset):
        """AWQModifier updates weight and bias via update_offload_parameter
        after grid search finds optimal smoothing scales."""
        before = _get_weight_snapshot(offloaded_model)

        recipe = [
            AWQModifier(duo_scaling=False, n_grid=3),
            QuantizationModifier(
                targets="Linear",
                scheme="W8A16",
                ignore=["lm_head"],
            ),
        ]

        oneshot(
            model=offloaded_model,
            dataset=dataset,
            recipe=recipe,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
        )

        after = _get_weight_snapshot(offloaded_model)
        _assert_weights_changed(before, after, ["q_proj", "k_proj", "v_proj"])

    def test_smoothquant_with_quantization(self, offloaded_model, dataset):
        """SmoothQuant followed by quantization: both transforms and quantization
        parameters must be persisted to the disk offload."""
        before = _get_weight_snapshot(offloaded_model)

        recipe = [
            SmoothQuantModifier(smoothing_strength=0.5),
            QuantizationModifier(
                targets="Linear",
                scheme="W8A16",
                ignore=["lm_head"],
            ),
        ]

        oneshot(
            model=offloaded_model,
            dataset=dataset,
            recipe=recipe,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
        )

        after = _get_weight_snapshot(offloaded_model)
        _assert_weights_changed(before, after, ["input_layernorm", "q_proj"])

        q_proj = offloaded_model.model.layers[0].self_attn.q_proj
        assert hasattr(
            q_proj, "weight_scale"
        ), "weight_scale not found after SmoothQuant + Quantization"


@pytest.mark.smoke
@pytest.mark.integration
class TestDiskOffloadSaveLoad:
    def test_save_and_reload(self, offloaded_model, dataset, tmp_path):
        """Verify that a disk-offloaded model can be quantized, saved, and
        reloaded with the quantization config intact."""
        oneshot(
            model=offloaded_model,
            dataset=dataset,
            recipe=QuantizationModifier(
                targets="Linear",
                scheme="W8A16",
                ignore=["lm_head"],
            ),
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
        )

        save_dir = tmp_path / "saved_model"
        offloaded_model.save_pretrained(save_dir, save_compressed=True)

        reloaded = AutoModelForCausalLM.from_pretrained(save_dir)
        quant_config = getattr_chain(
            reloaded, "config.quantization_config.quantization_config", None
        )
        assert quant_config is not None, "Quantization config not found after reload"
        assert "lm_head" in quant_config.ignore
