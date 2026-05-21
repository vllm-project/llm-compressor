"""
Integration test for the layerwise quantization pipeline.

Runs full layerwise AWQ + W4A16 quantization on Qwen/Qwen3-0.6B.
Verifies the pipeline produces valid compressed safetensors output.
"""

from pathlib import Path

import pytest
from safetensors import safe_open

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import AWQModifier
from tests.testing_utils import requires_gpu

MODEL_ID = "Qwen/Qwen3-0.6B"


@requires_gpu(1)
class TestLayerwisePipeline:
    """End-to-end tests for the layerwise quantization pipeline."""

    @pytest.mark.parametrize(
        "recipe,test_id",
        [
            pytest.param(
                [
                    AWQModifier(duo_scaling=False),
                    QuantizationModifier(scheme="W4A16"),
                ],
                "awq_w4a16",
                id="awq_w4a16",
            ),
            pytest.param(
                [QuantizationModifier(scheme="W4A16")],
                "w4a16_only",
                id="w4a16_only",
            ),
        ],
    )
    def test_layerwise_quantization_e2e(self, recipe, test_id, tmp_path):
        """
        Full layerwise quantization should produce valid compressed safetensors.

        This test verifies:
        1. The pipeline runs without error on a real model
        2. Output directory contains safetensors shards
        3. A safetensors index file is written
        4. Compressed weights are loadable
        """
        output_dir = str(tmp_path / test_id)

        oneshot(
            model=MODEL_ID,
            recipe=recipe,
            dataset="ultrachat_200k",
            splits={"calibration": "train_sft[:32]"},
            max_seq_length=512,
            num_calibration_samples=32,
            layerwise=True,
            output_dir=output_dir,
        )

        output_path = Path(output_dir)

        # Check safetensors index exists
        index_file = output_path / "model.safetensors.index.json"
        assert index_file.exists(), "Missing safetensors index file"

        # Check at least one shard exists
        shards = list(output_path.glob("model-*-of-*.safetensors"))
        assert len(shards) > 0, "No safetensors shards found"

        # Verify shards are loadable
        for shard in shards:
            with safe_open(str(shard), framework="pt") as f:
                keys = list(f.keys())
                assert len(keys) > 0, f"Shard {shard.name} has no keys"

        # Check config.json was saved
        assert (output_path / "config.json").exists(), "Missing config.json"

    def test_layerwise_matches_sequential_output_shape(self, tmp_path):
        """
        Layerwise and sequential pipelines should produce weights with
        the same shapes (validates that group quantization packs correctly).
        """
        recipe = [QuantizationModifier(scheme="W4A16")]

        layerwise_dir = str(tmp_path / "layerwise")
        oneshot(
            model=MODEL_ID,
            recipe=recipe,
            dataset="ultrachat_200k",
            splits={"calibration": "train_sft[:16]"},
            max_seq_length=256,
            num_calibration_samples=16,
            layerwise=True,
            output_dir=layerwise_dir,
        )

        sequential_dir = str(tmp_path / "sequential")
        oneshot(
            model=MODEL_ID,
            recipe=recipe,
            dataset="ultrachat_200k",
            splits={"calibration": "train_sft[:16]"},
            max_seq_length=256,
            num_calibration_samples=16,
            output_dir=sequential_dir,
        )

        # Compare weight shapes between both outputs
        layerwise_path = Path(layerwise_dir)
        sequential_path = Path(sequential_dir)

        lw_shards = sorted(layerwise_path.glob("model*.safetensors"))
        sq_shards = sorted(sequential_path.glob("model*.safetensors"))

        # Collect all weight shapes from both
        lw_shapes = {}
        for shard in lw_shards:
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():
                    lw_shapes[key] = f.get_tensor(key).shape

        sq_shapes = {}
        for shard in sq_shards:
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():
                    sq_shapes[key] = f.get_tensor(key).shape

        # All sequential keys should exist in layerwise output with same shape
        for key in sq_shapes:
            assert key in lw_shapes, f"Key {key} missing from layerwise output"
            assert lw_shapes[key] == sq_shapes[key], (
                f"Shape mismatch for {key}: "
                f"layerwise={lw_shapes[key]} vs sequential={sq_shapes[key]}"
            )
