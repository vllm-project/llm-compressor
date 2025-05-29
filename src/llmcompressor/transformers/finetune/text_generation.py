#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/huggingface/transformers
# vllm-project: no copyright


from compressed_tensors.utils.helpers import deprecated


@deprecated(
    message=(
        "`from llmcompressor.transformers import oneshot` is deprecated, "
        "please use `from llmcompressor import oneshot`."
    )
)
def oneshot(**kwargs) -> None:
    from llmcompressor import oneshot

    oneshot(**kwargs)


@deprecated(
    message=(
        "`from llmcompressor import train` is deprecated, "
        "please use `from llmcompressor import train`."
    )
)
def train(**kwargs):
    from llmcompressor import train

    train(**kwargs)


def apply(**kwargs):
    message = (
        "`from llmcompressor.transformers import apply, compress` is deprecated, "
        "please use `from llmcompressor import oneshot, train` "
        "for sequential stages."
    )
    raise ValueError(message)
