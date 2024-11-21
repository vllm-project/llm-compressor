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


def compressed_tensors_config_available():
    try:
        from transformers.utils.quantization_config import (  # noqa: F401
            CompressedTensorsConfig,
        )

        return True
    except ImportError:
        return False


def accelerate_availabe():
    try:
        import accelerate  # noqa: F401

        return True

    except ImportError:
        return False


_is_compressed_tensors_config_available = compressed_tensors_config_available()
_is_accelerate_available = accelerate_availabe()


def requires_hf_quantizer():
    return pytest.mark.skipif(
        not _is_compressed_tensors_config_available,
        reason="requires transformers>=4.45 to support CompressedTensorsHfQuantizer",
    )


def requires_accelerate():
    return pytest.mark.skipif(
        not _is_accelerate_available,
        reason="requires accelerate",
    )
