from unittest.mock import Mock, patch

import pytest
from torch.nn import Module

from llmcompressor.transformers.compression.sparsity_metadata_config import (
    SparsityConfigMetadata,
)

SPARSITY_CONFIG_LOCATION = (
    "llmcompressor.transformers.compression.sparsity_metadata_config"
)


# Mock classes and functions
class MockSparsityStructure:
    TWO_FOUR = Mock(value="2:4")


class MockQuantizationType:
    INT = Mock(value="int")
    FLOAT = Mock(value="float")


class MockSparsityConfigMetadata:
    @staticmethod
    def infer_sparsity_structure(model):
        return model.sparsity_structure


def mock_is_model_quantized(model):
    return model.is_quantized


# Mock model class
class MockModel(Module):
    def __init__(
        self, sparsity_structure=None, is_quantized=False, quantization_scheme=None
    ):
        super().__init__()
        self.sparsity_structure = sparsity_structure
        self.is_quantized = is_quantized
        self.quantization_scheme = quantization_scheme

    def named_modules(self):
        yield "mock_submodule", self


# Fixtures
@pytest.fixture
def models():
    return {
        "non_sparse": MockModel(sparsity_structure=None),
        "non_24_sparse": MockModel(sparsity_structure="unstructured"),
        "non_quantized_24_sparse": MockModel(
            sparsity_structure=MockSparsityStructure.TWO_FOUR.value, is_quantized=False
        ),
        "quantized_24_sparse_supported": MockModel(
            sparsity_structure=MockSparsityStructure.TWO_FOUR.value,
            is_quantized=True,
            # W8A8
            quantization_scheme=Mock(
                weights=Mock(num_bits=8, type=MockQuantizationType.FLOAT.value),
                input_activations=Mock(
                    num_bits=8, type=MockQuantizationType.FLOAT.value
                ),
            ),
        ),
        "quantized_24_sparse_unsupported": MockModel(
            sparsity_structure=MockSparsityStructure.TWO_FOUR.value,
            is_quantized=True,
            # W4A8
            quantization_scheme=Mock(
                weights=Mock(num_bits=4, type=MockQuantizationType.INT.value),
                input_activations=Mock(
                    num_bits=8, type=MockQuantizationType.FLOAT.value
                ),
            ),
        ),
    }


@pytest.mark.usefixtures("models")
class TestSparse24BitmaskSupport:
    @pytest.fixture(autouse=True)
    def setup_mocks(self, request):
        patchers = [
            patch(
                f"{SPARSITY_CONFIG_LOCATION}"
                ".SparsityConfigMetadata.infer_sparsity_structure",
                side_effect=MockSparsityConfigMetadata.infer_sparsity_structure,
            ),
            patch(
                f"{SPARSITY_CONFIG_LOCATION}.is_model_quantized",
                side_effect=mock_is_model_quantized,
            ),
        ]
        for patcher in patchers:
            patcher.start()
            request.addfinalizer(patcher.stop)  # for cleanup

    @pytest.mark.parametrize(
        "model_key, expected",
        [
            ("non_sparse", False),
            ("non_24_sparse", False),
            ("non_quantized_24_sparse", True),
            ("quantized_24_sparse_supported", True),
            ("quantized_24_sparse_unsupported", False),
        ],
    )
    def test_sparse24_bitmask_support(self, models, model_key, expected):
        model = models[model_key]
        result = SparsityConfigMetadata.is_sparse24_bitmask_supported(model)
        assert result == expected
