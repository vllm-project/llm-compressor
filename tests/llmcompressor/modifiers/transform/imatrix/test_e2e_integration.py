import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "nm-testing/tinysmokellama-3.2"
DATASET = "open_platypus"
NUM_CALIB_SAMPLES = 4
MAX_SEQ_LEN = 128


def _get_linear_layer_names(model, ignore=None):
    """Return names of all nn.Linear modules not in the ignore list."""
    ignore = ignore or []
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if not any(pattern in name for pattern in ignore):
                names.append(name)
    return names


def _get_module_by_name(model, name):
    """Retrieve a submodule by dotted name."""
    parts = name.split(".")
    m = model
    for part in parts:
        m = getattr(m, part)
    return m


class TestIMatrixObserverIntegration:
    """imatrix_mse observer end-to-end."""

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_pipeline_produces_quantized_model(self):
        """Observer collects importance and produces a quantized model."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

        recipe = [
            QuantizationModifier(
                config_groups={
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": True,
                            "strategy": "group",
                            "group_size": 32,
                            "observer": "imatrix_mse",
                            "observer_kwargs": {
                                "norm": 2.4,
                                "maxshrink": 0.7,
                            },
                        },
                    }
                },
                ignore=["lm_head"],
            ),
        ]

        oneshot(
            model=model,
            dataset=DATASET,
            splits="train[:5%]",
            recipe=recipe,
            num_calibration_samples=NUM_CALIB_SAMPLES,
            max_seq_length=MAX_SEQ_LEN,
        )

        targeted_names = _get_linear_layer_names(model, ignore=["lm_head"])
        for name in targeted_names:
            mod = _get_module_by_name(model, name)
            assert hasattr(mod, "weight_scale"), f"{name} should be quantized"
            assert not hasattr(mod, "weight_observer")

        # Hooks should be cleaned up
        total_hooks = sum(len(m._forward_pre_hooks) for m in model.modules())
        assert (
            total_hooks == 0
        ), f"Expected 0 forward pre-hooks after completion, found {total_hooks}"

        # Verify imatrix actually affected quantization by comparing scales
        # with a run using uniform memoryless_mse. Same kwargs, same calibration;
        # the only difference is importance weighting.
        imatrix_scales = {
            name: _get_module_by_name(model, name).weight_scale.clone()
            for name in targeted_names
            if hasattr(_get_module_by_name(model, name), "weight_scale")
        }

        model_uniform = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        recipe_uniform = [
            QuantizationModifier(
                config_groups={
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": True,
                            "strategy": "group",
                            "group_size": 32,
                            "observer": "memoryless_mse",
                            "observer_kwargs": {
                                "norm": 2.4,
                                "maxshrink": 0.7,
                            },
                        },
                    }
                },
                ignore=["lm_head"],
            ),
        ]
        oneshot(
            model=model_uniform,
            dataset=DATASET,
            splits="train[:5%]",
            recipe=recipe_uniform,
            num_calibration_samples=NUM_CALIB_SAMPLES,
            max_seq_length=MAX_SEQ_LEN,
        )

        differs = False
        for name in imatrix_scales:
            mod_uniform = _get_module_by_name(model_uniform, name)
            if hasattr(mod_uniform, "weight_scale"):
                if not torch.equal(imatrix_scales[name], mod_uniform.weight_scale):
                    differs = True
                    break

        assert differs, (
            "imatrix_mse produced identical scales to memoryless_mse — "
            "importance was not collected"
        )

    def test_pipeline_with_regex_targets(self):
        """Observer supports regex targets for specific attention projections."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

        recipe = [
            QuantizationModifier(
                config_groups={
                    "group_0": {
                        "targets": ["re:.*self_attn.(q|k|v)_proj$"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": True,
                            "strategy": "group",
                            "group_size": 32,
                            "observer": "imatrix_mse",
                        },
                    }
                },
                ignore=["lm_head"],
            ),
        ]

        oneshot(
            model=model,
            dataset=DATASET,
            splits="train[:5%]",
            recipe=recipe,
            num_calibration_samples=NUM_CALIB_SAMPLES,
            max_seq_length=MAX_SEQ_LEN,
        )

        for name, module in model.named_modules():
            assert not hasattr(
                module, "weight_observer"
            ), f"{name} should not retain its weight observer after finalization"

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_observer_completes_without_separate_gatherer(self):
        """imatrix_mse runs directly through QuantizationModifier."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

        recipe = [
            QuantizationModifier(
                config_groups={
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": True,
                            "strategy": "group",
                            "group_size": 32,
                            "observer": "imatrix_mse",
                        },
                    }
                },
                ignore=["lm_head"],
            ),
        ]

        oneshot(
            model=model,
            dataset=DATASET,
            splits="train[:5%]",
            recipe=recipe,
            num_calibration_samples=NUM_CALIB_SAMPLES,
            max_seq_length=MAX_SEQ_LEN,
        )
