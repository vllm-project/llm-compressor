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

import pytest
from compressed_tensors.config import SparsityStructure


def test_sparsity_structure_valid_cases():
    assert (
        SparsityStructure("2:4") == SparsityStructure.TWO_FOUR
    ), "Failed to match '2:4' with TWO_FOUR"
    assert (
        SparsityStructure("unstructured") == SparsityStructure.UNSTRUCTURED
    ), "Failed to match 'unstructured' with UNSTRUCTURED"
    assert (
        SparsityStructure("UNSTRUCTURED") == SparsityStructure.UNSTRUCTURED
    ), "Failed to match 'UNSTRUCTURED' with UNSTRUCTURED"
    assert (
        SparsityStructure(None) == SparsityStructure.UNSTRUCTURED
    ), "Failed to match None with UNSTRUCTURED"


def test_sparsity_structure_invalid_case():
    with pytest.raises(ValueError, match="invalid is not a valid SparsityStructure"):
        SparsityStructure("invalid")


def test_sparsity_structure_case_insensitivity():
    assert (
        SparsityStructure("2:4") == SparsityStructure.TWO_FOUR
    ), "Failed to match '2:4' with TWO_FOUR"
    assert (
        SparsityStructure("2:4".upper()) == SparsityStructure.TWO_FOUR
    ), "Failed to match '2:4'.upper() with TWO_FOUR"
    assert (
        SparsityStructure("unstructured".upper()) == SparsityStructure.UNSTRUCTURED
    ), "Failed to match 'unstructured'.upper() with UNSTRUCTURED"
    assert (
        SparsityStructure("UNSTRUCTURED".lower()) == SparsityStructure.UNSTRUCTURED
    ), "Failed to match 'UNSTRUCTURED'.lower() with UNSTRUCTURED"


def test_sparsity_structure_default_case():
    assert (
        SparsityStructure(None) == SparsityStructure.UNSTRUCTURED
    ), "Failed to match None with UNSTRUCTURED"
