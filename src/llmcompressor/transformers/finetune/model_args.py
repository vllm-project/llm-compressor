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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from
    """

    model: str = field(
        metadata={
            "help": (
                "A pretrained model or a string as a path to pretrained model, "
                "sparsezoo stub, or model identifier from huggingface.co/models."
            )
        },
    )
    distill_teacher: Optional[str] = field(
        default=None,
        metadata={
            "help": "Teacher model (a trained text generation model)",
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained data from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizers. Default True"},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use "
            "(can be a branch name, tag name or commit id)"
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use token generated when running `transformers-cli login` "
            "(necessary to use this script with private models)"
        },
    )
    precision: str = field(
        default="auto",
        metadata={"help": "Precision to cast model weights to, default to auto"},
    )

    tie_word_embeddings: bool = field(
        default=False,
        metadata={
            "help": "Whether the model's input and output word embeddings "
            "should be tied. Note that this is only relevant if the "
            "model has a output word embedding layer."
        },
    )
