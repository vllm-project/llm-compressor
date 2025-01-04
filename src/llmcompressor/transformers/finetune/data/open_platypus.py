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


@TextGenerationDataset.register(name="open_platypus")
class OpenPlatypusDataset(TextGenerationDataset):
    """
    Child text generation class for the Open Platypus dataset

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    ALPACA_TEMPLATE = {
        "prompt_input": "Below is an instruction that describes a task, paired with an "
        "input that provides further context. Write a response that appropriately "
        "completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n"
        "{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a "
        "response that appropriately completes the request.\n\n### Instruction:\n{"
        "instruction}\n\n### Response:\n",
    }

    def __init__(self, data_args: "DataArgs", split: str, processor: Processor):
        data_args = deepcopy(data_args)
        data_args.dataset = "garage-bAInd/Open-Platypus"
        data_args.text_column = "text"
        super().__init__(data_args=data_args, split=split, processor=processor)

    def dataset_template(self, sample):
        if "input" in sample and sample["input"] != "":
            prompt = self.ALPACA_TEMPLATE["prompt_input"].format(
                instruction=sample["instruction"], input=sample["input"]
            )
        else:
            prompt = self.ALPACA_TEMPLATE["prompt_no_input"].format(
                instruction=sample["instruction"]
            )

        text = prompt
        if "output" in sample:
            text += sample["output"]

        return {
            "text": text,
            self.PROMPT_KEY: prompt,
        }
