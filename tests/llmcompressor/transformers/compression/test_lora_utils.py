"""
Tests for LoRA compatibility utilities.
"""

import pytest
import torch

from llmcompressor.transformers.compression.lora_utils import (
    get_lora_metadata,
    materialize_weights_for_lora,
    unpack_int4_for_lora,
    unpack_int4_weights,
)


class TestUnpackINT4Weights:
    """Test unpacking of INT4 quantized weights."""

    def test_unpack_int4_per_channel(self):
        """Test unpacking with per-channel quantization."""
        # Create mock packed weights: [4, 2] -> unpacks to [4, 4]
        packed = torch.tensor(
            [
                [0x12, 0x34],  # Unpacks to [2, 1, 4, 3]
                [0x56, 0x78],  # Unpacks to [6, 5, 8, 7]
                [0x9A, 0xBC],  # Unpacks to [10, 9, 12, 11]
                [0xDE, 0xF0],  # Unpacks to [14, 13, 0, 15]
            ],
            dtype=torch.uint8,
        )

        # Per-channel scales: one per output channel
        scales = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)

        unpacked = unpack_int4_weights(packed, scales, zero_points=None)

        assert unpacked.shape == (4, 4)
        assert unpacked.dtype == torch.float16

        # Verify unpacking: INT4 values are [0-15] -> [-8, 7] after signed conversion
        # First row: [0x2, 0x1, 0x4, 0x3] -> [-6, -7, -4, -5] * 1.0
        expected_first_row = torch.tensor([-6.0, -7.0, -4.0, -5.0], dtype=torch.float16)
        torch.testing.assert_close(
            unpacked[0], expected_first_row, rtol=1e-3, atol=1e-3
        )

    def test_unpack_int4_grouped(self):
        """Test unpacking with grouped quantization."""
        # Create mock packed weights: [2, 4] -> unpacks to [2, 8]
        # With group_size=4, we have 2 groups per row
        packed = torch.randint(0, 255, (2, 4), dtype=torch.uint8)

        # Grouped scales: [out_features, num_groups]
        # For in_features=8 and group_size=4, num_groups=2
        scales = torch.tensor(
            [
                [1.0, 2.0],  # Two groups for first output channel
                [3.0, 4.0],  # Two groups for second output channel
            ],
            dtype=torch.float16,
        )

        unpacked = unpack_int4_weights(packed, scales, zero_points=None, group_size=4)

        assert unpacked.shape == (2, 8)
        assert unpacked.dtype == torch.float16

        # Check that first 4 elements use scale 1.0 and next 4 use scale 2.0
        # We won't check exact values since input is random, just shapes

    def test_unpack_int4_with_zero_point(self):
        """Test unpacking with asymmetric quantization (zero points)."""
        packed = torch.randint(0, 255, (2, 2), dtype=torch.uint8)
        scales = torch.tensor([1.0, 2.0], dtype=torch.float16)
        zero_points = torch.tensor([0.0, 1.0], dtype=torch.float16)

        unpacked = unpack_int4_weights(packed, scales, zero_points=zero_points)

        assert unpacked.shape == (2, 4)
        assert unpacked.dtype == torch.float16

    def test_unpack_int4_invalid_dtype(self):
        """Test that non-uint8 input raises error."""
        packed = torch.randint(0, 127, (2, 2), dtype=torch.int8)
        scales = torch.ones(2, dtype=torch.float16)

        with pytest.raises(ValueError, match="must be uint8"):
            unpack_int4_weights(packed, scales)

    def test_unpack_int4_output_dtype(self):
        """Test unpacking with different output dtypes."""
        packed = torch.randint(0, 255, (2, 2), dtype=torch.uint8)
        scales = torch.ones(2, dtype=torch.float16)

        # Test bfloat16 output
        unpacked_bf16 = unpack_int4_weights(packed, scales, output_dtype=torch.bfloat16)
        assert unpacked_bf16.dtype == torch.bfloat16

        # Test float32 output
        unpacked_fp32 = unpack_int4_weights(packed, scales, output_dtype=torch.float32)
        assert unpacked_fp32.dtype == torch.float32


class TestUnpackINT4ForLoRA:
    """Test module-level unpacking for LoRA."""

    def test_unpack_module_with_packed_weights(self):
        """Test unpacking from a module with packed weights."""

        class MockQuantizedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight_packed", torch.randint(0, 255, (4, 2), dtype=torch.uint8)
                )
                self.register_buffer("weight_scale", torch.ones(4, dtype=torch.float16))

        module = MockQuantizedModule()
        unpacked = unpack_int4_for_lora(module)

        assert unpacked is not None
        assert unpacked.shape == (4, 4)  # Unpacked from (4, 2)
        assert unpacked.dtype == torch.float16

    def test_unpack_module_without_quantization(self):
        """Test that regular modules return None."""
        module = torch.nn.Linear(4, 4)
        unpacked = unpack_int4_for_lora(module)
        assert unpacked is None

    def test_unpack_module_missing_scales(self):
        """Test that modules without scales return None with warning."""

        class MockInvalidModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight_packed", torch.randint(0, 255, (4, 2), dtype=torch.uint8)
                )
                # Missing weight_scale!

        module = MockInvalidModule()
        unpacked = unpack_int4_for_lora(module)
        assert unpacked is None


class TestMaterializeWeightsForLoRA:
    """Test materializing FP weights for entire models."""

    def test_materialize_all_modules(self):
        """Test materializing weights for all quantized modules."""

        class MockQuantizedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = self._create_quant_module()
                self.v_proj = self._create_quant_module()
                self.fc = torch.nn.Linear(4, 4)  # Regular module

            def _create_quant_module(self):
                module = torch.nn.Module()
                module.register_buffer(
                    "weight_packed", torch.randint(0, 255, (4, 2), dtype=torch.uint8)
                )
                module.register_buffer(
                    "weight_scale", torch.ones(4, dtype=torch.float16)
                )
                return module

        model = MockQuantizedModel()
        unpacked_weights = materialize_weights_for_lora(model, inplace=False)

        # Should have unpacked 2 modules (q_proj and v_proj)
        assert len(unpacked_weights) == 2
        assert "q_proj" in unpacked_weights
        assert "v_proj" in unpacked_weights

        # Modules should have weight_lora buffer
        assert hasattr(model.q_proj, "weight_lora")
        assert hasattr(model.v_proj, "weight_lora")

        # Original packed weights should still exist
        assert hasattr(model.q_proj, "weight_packed")

    def test_materialize_target_modules(self):
        """Test materializing only specific target modules."""

        class MockQuantizedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = self._create_quant_module()
                self.k_proj = self._create_quant_module()
                self.v_proj = self._create_quant_module()

            def _create_quant_module(self):
                module = torch.nn.Module()
                module.register_buffer(
                    "weight_packed", torch.randint(0, 255, (4, 2), dtype=torch.uint8)
                )
                module.register_buffer(
                    "weight_scale", torch.ones(4, dtype=torch.float16)
                )
                return module

        model = MockQuantizedModel()
        unpacked_weights = materialize_weights_for_lora(
            model, target_modules=["q_proj", "v_proj"], inplace=False
        )

        # Should only unpack q_proj and v_proj, not k_proj
        assert len(unpacked_weights) == 2
        assert "q_proj" in unpacked_weights
        assert "v_proj" in unpacked_weights
        assert "k_proj" not in unpacked_weights

    def test_materialize_inplace(self):
        """Test in-place replacement of packed weights."""

        class MockQuantizedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Module()
                self.q_proj.register_buffer(
                    "weight_packed", torch.randint(0, 255, (4, 2), dtype=torch.uint8)
                )
                self.q_proj.register_buffer(
                    "weight_scale", torch.ones(4, dtype=torch.float16)
                )

        model = MockQuantizedModel()
        materialize_weights_for_lora(model, inplace=True)

        # Packed weights should be removed
        assert not hasattr(model.q_proj, "weight_packed")

        # Should have regular weight buffer instead
        assert hasattr(model.q_proj, "weight")
        assert model.q_proj.weight.dtype == torch.float16


class TestGetLoRAMetadata:
    """Test extraction of LoRA metadata."""

    def test_get_metadata_from_quantized_model(self):
        """Test extracting metadata from a quantized model."""

        class MockQuantizedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.model.layers[0].self_attn = torch.nn.Module()
                self.model.layers[0].self_attn.q_proj = self._create_quant_module()
                self.model.layers[0].self_attn.v_proj = self._create_quant_module()
                self.model.layers[0].mlp = torch.nn.Module()
                self.model.layers[0].mlp.gate_proj = self._create_quant_module()

            def _create_quant_module(self):
                module = torch.nn.Module()
                module.register_buffer(
                    "weight_packed", torch.randint(0, 255, (4, 2), dtype=torch.uint8)
                )
                module.register_buffer(
                    "weight_scale", torch.randn(4, 2, dtype=torch.float16)
                )
                return module

        model = MockQuantizedModel()
        metadata = get_lora_metadata(model)

        assert metadata["num_quantized_modules"] == 3
        assert metadata["lora_compatible"] is True
        assert len(metadata["quantized_modules"]) == 3

        # Should suggest common LoRA targets
        assert "q_proj" in metadata["suggested_lora_targets"]
        assert "v_proj" in metadata["suggested_lora_targets"]
        assert "gate_proj" in metadata["suggested_lora_targets"]

    def test_get_metadata_from_dense_model(self):
        """Test extracting metadata from a non-quantized model."""
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 4),
        )

        metadata = get_lora_metadata(model)

        assert metadata["num_quantized_modules"] == 0
        assert metadata["lora_compatible"] is True
        assert len(metadata["quantized_modules"]) == 0
