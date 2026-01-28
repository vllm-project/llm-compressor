import pytest
import torch

from llmcompressor.utils.block_quant_padding import (
    calculate_block_padding,
    needs_block_padding,
    pad_weight_for_block_quant,
)


@pytest.mark.unit
class TestCalculateBlockPadding:
    """Tests for calculate_block_padding function."""

    @pytest.mark.parametrize(
        "shape,block_structure,expected",
        [
            # No padding needed - divisible dimensions
            ((128, 128), [128, 128], (0, 0)),
            ((256, 256), [128, 128], (0, 0)),
            ((256, 128), [128, 128], (0, 0)),
            # Padding needed for out_features only
            ((10944, 2048), [128, 128], (64, 0)),  # 10944 % 128 = 64, need 64 more
            ((100, 128), [128, 128], (28, 0)),  # 100 % 128 = 100, need 28 more
            # Padding needed for in_features only
            ((128, 10944), [128, 128], (0, 64)),
            ((128, 100), [128, 128], (0, 28)),
            # Padding needed for both dimensions
            ((10944, 10944), [128, 128], (64, 64)),
            ((100, 100), [128, 128], (28, 28)),
            # Different block sizes
            ((100, 100), [64, 64], (28, 28)),  # 100 % 64 = 36, need 28 more
            ((100, 100), [32, 32], (28, 28)),  # 100 % 32 = 4, need 28 more
        ],
    )
    def test_calculate_block_padding(self, shape, block_structure, expected):
        result = calculate_block_padding(shape, block_structure)
        assert result == expected


@pytest.mark.unit
class TestNeedsBlockPadding:
    """Tests for needs_block_padding function."""

    @pytest.mark.parametrize(
        "shape,block_structure,expected",
        [
            # No padding needed
            ((128, 128), [128, 128], False),
            ((256, 256), [128, 128], False),
            # Padding needed
            ((10944, 2048), [128, 128], True),  # DeepSeek-V2 intermediate_size
            ((100, 100), [128, 128], True),
            ((127, 128), [128, 128], True),
            ((128, 127), [128, 128], True),
        ],
    )
    def test_needs_block_padding(self, shape, block_structure, expected):
        result = needs_block_padding(shape, block_structure)
        assert result == expected


@pytest.mark.unit
class TestPadWeightForBlockQuant:
    """Tests for pad_weight_for_block_quant function."""

    def test_no_padding_needed(self):
        """Test that weights divisible by block size are not padded."""
        weight = torch.randn(256, 256)
        block_structure = [128, 128]

        padded_weight, original_shape = pad_weight_for_block_quant(
            weight, block_structure
        )

        assert original_shape is None
        assert padded_weight.shape == weight.shape
        assert torch.equal(padded_weight, weight)

    def test_padding_out_features(self):
        """Test padding when out_features is not divisible by block_n."""
        weight = torch.randn(10944, 2048)
        block_structure = [128, 128]

        padded_weight, original_shape = pad_weight_for_block_quant(
            weight, block_structure
        )

        assert original_shape == (10944, 2048)
        # 10944 + 64 = 11008, which is divisible by 128
        assert padded_weight.shape == (11008, 2048)
        # Original values should be preserved
        assert torch.equal(padded_weight[:10944, :], weight)
        # Padding should be zeros
        assert torch.all(padded_weight[10944:, :] == 0)

    def test_padding_in_features(self):
        """Test padding when in_features is not divisible by block_k."""
        weight = torch.randn(2048, 10944)
        block_structure = [128, 128]

        padded_weight, original_shape = pad_weight_for_block_quant(
            weight, block_structure
        )

        assert original_shape == (2048, 10944)
        # 10944 + 64 = 11008, which is divisible by 128
        assert padded_weight.shape == (2048, 11008)
        # Original values should be preserved
        assert torch.equal(padded_weight[:, :10944], weight)
        # Padding should be zeros
        assert torch.all(padded_weight[:, 10944:] == 0)

    def test_padding_both_dimensions(self):
        """Test padding when both dimensions need padding."""
        weight = torch.randn(10944, 10944)
        block_structure = [128, 128]

        padded_weight, original_shape = pad_weight_for_block_quant(
            weight, block_structure
        )

        assert original_shape == (10944, 10944)
        assert padded_weight.shape == (11008, 11008)
        # Original values should be preserved
        assert torch.equal(padded_weight[:10944, :10944], weight)
        # Padding should be zeros
        assert torch.all(padded_weight[10944:, :] == 0)
        assert torch.all(padded_weight[:, 10944:] == 0)

    def test_preserves_dtype(self):
        """Test that padding preserves the weight dtype."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            weight = torch.randn(100, 100, dtype=dtype)
            block_structure = [128, 128]

            padded_weight, _ = pad_weight_for_block_quant(weight, block_structure)

            assert padded_weight.dtype == dtype

    def test_preserves_device(self):
        """Test that padding preserves the weight device."""
        weight = torch.randn(100, 100)
        block_structure = [128, 128]

        padded_weight, _ = pad_weight_for_block_quant(weight, block_structure)

        assert padded_weight.device == weight.device

    def test_invalid_dimension_raises_error(self):
        """Test that non-2D tensors raise an error."""
        weight_3d = torch.randn(10, 10, 10)
        block_structure = [128, 128]

        with pytest.raises(ValueError, match="Expected 2D weight tensor"):
            pad_weight_for_block_quant(weight_3d, block_structure)

    def test_different_block_sizes(self):
        """Test padding with different block sizes."""
        weight = torch.randn(100, 100)

        # Block size 64x64
        padded_64, original_64 = pad_weight_for_block_quant(weight, [64, 64])
        assert original_64 == (100, 100)
        assert padded_64.shape == (128, 128)  # 100 -> 128 (nearest multiple of 64)

        # Block size 32x32
        padded_32, original_32 = pad_weight_for_block_quant(weight, [32, 32])
        assert original_32 == (100, 100)
        assert padded_32.shape == (128, 128)  # 100 -> 128 (nearest multiple of 32)
