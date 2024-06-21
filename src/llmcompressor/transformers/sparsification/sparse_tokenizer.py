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

import os

from sparsezoo import Model
from transformers import AutoTokenizer

from llmcompressor.transformers.utils.helpers import POSSIBLE_TOKENIZER_FILES
from llmcompressor.utils.fsdp.context import main_process_first_context

__all__ = ["SparseAutoTokenizer"]


class SparseAutoTokenizer(AutoTokenizer):
    """
    SparseML wrapper for the AutoTokenizer class
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        A wrapper around the AutoTokenizer.from_pretrained method that
        enables the loading of tokenizer from SparseZoo stubs

        If a SparseZoo stub is passed, the all the available tokenizer
        files are downloaded and the path to the directory containing the
        files is passed to the AutoTokenizer.from_pretrained method

        :param pretrained_model_name_or_path: the name of or path to the model to load
        :return tokenizer: the loaded tokenizer from pretrained
        """
        if str(pretrained_model_name_or_path).startswith("zoo:"):
            with main_process_first_context():
                model = Model(pretrained_model_name_or_path)
                for file_name in POSSIBLE_TOKENIZER_FILES:
                    # go over all the possible tokenizer files
                    # and if detected, download them
                    file = model.deployment.get_file(file_name)
                    if file is not None:
                        tokenizer_file = file
                        tokenizer_file.download()
                pretrained_model_name_or_path = os.path.dirname(tokenizer_file.path)
        return super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
