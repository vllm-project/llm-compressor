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

from unittest.mock import patch

import pytest
from compressed_tensors.utils.safetensors_load import get_nested_weight_mappings


mock_weight_mappings = {
    "layer1.weight": "file1",
    "layer1.bias": "file2",
    "layer2.weight": "file3",
    "layer2.bias": "file4",
    "layer3.weight": "file5",
}


@pytest.fixture
def mock_get_weight_mappings():
    with patch(
        "compressed_tensors.utils.safetensors_load.get_weight_mappings",
        return_value=mock_weight_mappings,
    ):
        yield


@pytest.mark.usefixtures("mock_get_weight_mappings")
class TestGetNestedWeightMappings:
    """
    Tests for the get_nested_weight_mappings function
    in different scenarios, such as single and multiple
    parameters to nest, and returning other parameters
    """

    def test_single_param(self):
        params_to_nest = ["weight"]
        result = get_nested_weight_mappings("dummy_path", params_to_nest)
        expected = {
            "layer1": {"weight": "file1"},
            "layer2": {"weight": "file3"},
            "layer3": {"weight": "file5"},
        }
        assert result == expected

    def test_multiple_params(self):
        params_to_nest = ["weight", "bias"]
        result = get_nested_weight_mappings("dummy_path", params_to_nest)
        expected = {
            "layer1": {"weight": "file1", "bias": "file2"},
            "layer2": {"weight": "file3", "bias": "file4"},
            "layer3": {"weight": "file5"},
        }
        assert result == expected

    def test_return_other_params(self):
        params_to_nest = ["weight"]
        result, other_params = get_nested_weight_mappings(
            "dummy_path", params_to_nest, return_unmatched_params=True
        )
        expected_nested = {
            "layer1": {"weight": "file1"},
            "layer2": {"weight": "file3"},
            "layer3": {"weight": "file5"},
        }
        expected_other = {
            "layer1.bias": "file2",
            "layer2.bias": "file4",
        }
        assert result == expected_nested
        assert other_params == expected_other
