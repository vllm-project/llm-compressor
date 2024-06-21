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


from sparsezoo import Model
from transformers import AutoConfig

__all__ = ["SparseAutoConfig"]


class SparseAutoConfig(AutoConfig):
    """
    SparseML wrapper for the AutoConfig class
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        A wrapper around the AutoConfig.from_pretrained method that
        enables the loading of configs from SparseZoo stubs

        If a SparseZoo stub is passed, the all the available config
        file is passed to the AutoTokenizer.from_pretrained method

        :param pretrained_model_name_or_path: the name of or path to the model to load
        :return tokenizer: the loaded tokenizer from pretrained
        """
        if str(pretrained_model_name_or_path).startswith("zoo:"):
            model = Model(pretrained_model_name_or_path)
            config = model.training.get_file(file_name="config.json")
            if config is None:
                raise ValueError(
                    "Could not find config.json for stub: "
                    f"{pretrained_model_name_or_path}"
                )
            pretrained_model_name_or_path = config.path

        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
