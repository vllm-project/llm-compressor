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
from sparsification_config import (
    BitmaskCompressor,
    BitmaskConfig,
    CompressionConfig,
    DenseCompressor,
    DenseSparsityConfig,
    ModelCompressor,
)


@pytest.mark.parametrize(
    "name,type",
    [
        ["sparse_bitmask", BitmaskConfig],
        ["dense_sparsity", DenseSparsityConfig],
    ],
)
def test_configs(name, type):
    config = CompressionConfig.load_from_registry(name)
    assert isinstance(config, type)
    assert config.format == name


@pytest.mark.parametrize(
    "name,type",
    [["sparse_bitmask", BitmaskCompressor], ["dense_sparsity", DenseCompressor]],
)
def test_compressors(name, type):
    compressor = ModelCompressor.load_from_registry(
        name, config=CompressionConfig(format="none")
    )
    assert isinstance(compressor, type)
    assert isinstance(compressor.config, CompressionConfig)
    assert compressor.config.format == "none"
