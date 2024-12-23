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

from loguru import logger

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.typing import Processor

if TYPE_CHECKING:
    from llmcompressor.transformers import DataTrainingArguments as DataArgs


@TextGenerationDataset.register(name="ultrachat_200k")
class UltraChatDataset(TextGenerationDataset):
    """
    Child text generation class for the Ultra Chat 200k dataset

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    DEFAULT_CHAT_TEMPLATE = (
        "{% for message in messages %}\n"
        "{% if message['role'] == 'user' %}\n"
        "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
        "{% elif message['role'] == 'system' %}\n"
        "{{ '<|system|>\n' + message['content'] + eos_token }}\n"
        "{% elif message['role'] == 'assistant' %}\n"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
        "{% endif %}\n"
        "{% if loop.last and add_generation_prompt %}\n"
        "{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    )

    def __init__(self, data_args: "DataArgs", split: str, processor: Processor):
        data_args = deepcopy(data_args)
        data_args.dataset = "HuggingFaceH4/ultrachat_200k"
        data_args.text_column = "messages"

        if split in ["train", "test"]:
            split += "_sft"

        super().__init__(data_args=data_args, split=split, processor=processor)

        if (
            self.tokenizer is not None
            and getattr(self.tokenizer, "chat_template", None) is None
        ):
            # note that since tokenizer is a member of processor,
            # this change affects processor.apply_chat_template
            self.tokenizer.chat_template = self.DEFAULT_CHAT_TEMPLATE
            logger.warning(
                "tokenizer.chat_template is not set, using default chat template for "
                f"{self.__class__.__name__}"
            )

    def dataset_template(self, sample):
        messages = sample["messages"]
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})

        return {
            "text": self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }
