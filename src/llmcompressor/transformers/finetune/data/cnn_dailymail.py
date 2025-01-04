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
from copy import deepcopy
from typing import TYPE_CHECKING

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.typing import Processor

if TYPE_CHECKING:
    from llmcompressor.transformers import DataTrainingArguments as DataArgs


@TextGenerationDataset.register(name="cnn_dailymail")
class CNNDailyMailDataset(TextGenerationDataset):
    """
    Text generation class for the CNN/DailyMail dataset

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    SAMPLE_TEMPLATE = "Article:\n{article}\n\n### Summarization:\n{highlights}\n"

    def __init__(self, data_args: "DataArgs", split: str, processor: Processor):
        data_args = deepcopy(data_args)
        data_args.dataset = "cnn_dailymail"
        data_args.dataset_config_name = "3.0.0"

        super().__init__(data_args=data_args, split=split, processor=processor)

    def dataset_template(self, sample):
        return {
            "text": self.SAMPLE_TEMPLATE.format(
                article=sample["article"], highlights=sample["highlights"]
            )
        }
