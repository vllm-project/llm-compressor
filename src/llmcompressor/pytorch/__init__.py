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

"""
Functionality for working with and sparsifying Models in the PyTorch framework
"""

import os
import warnings

from packaging import version

try:
    import torch

    _PARSED_TORCH_VERSION = version.parse(torch.__version__)

    if _PARSED_TORCH_VERSION.major >= 2:
        torch_compile_func = torch.compile

        def raise_torch_compile_warning(*args, **kwargs):
            warnings.warn(
                "torch.compile is not supported by llmcompressor for torch 2.0.x"
            )
            return torch_compile_func(*args, **kwargs)

        torch.compile = raise_torch_compile_warning

    _BYPASS = bool(int(os.environ.get("NM_BYPASS_TORCH_VERSION", "0")))
    if _PARSED_TORCH_VERSION.major == 1 and _PARSED_TORCH_VERSION.minor in [10, 11]:
        if not _BYPASS:
            raise RuntimeError(
                "llmcompressor does not support torch==1.10.* or 1.11.*. "
                f"Found torch version {torch.__version__}.\n\n"
                "To bypass this error, set environment variable "
                "`NM_BYPASS_TORCH_VERSION` to '1'.\n\n"
                "Bypassing may result in errors or "
                "incorrect behavior, so set at your own risk."
            )
        else:
            warnings.warn(
                "llmcompressor quantized onnx export does not work "
                "with torch==1.10.* or 1.11.*"
            )
except ImportError:
    pass

# flake8: noqa
