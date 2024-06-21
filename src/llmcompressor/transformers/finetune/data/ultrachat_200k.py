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
from typing import Optional

from llmcompressor.transformers.finetune.data import TextGenerationDataset


@TextGenerationDataset.register(name="ultrachat_200k")
class UltraChatDataset(TextGenerationDataset):
    """
    Child text generation class for the Ultra Chat 200k dataset

    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param tokenizer: tokenizer to use on dataset
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

    def __init__(self, data_args, split, tokenizer):
        data_args = deepcopy(data_args)
        data_args.dataset = "HuggingFaceH4/ultrachat_200k"

        if split in ["train", "test"]:
            split += "_sft"

        super().__init__(
            text_column="messages",
            data_args=data_args,
            split=split,
            tokenizer=tokenizer,
        )

        if (
            not hasattr(self.tokenizer, "chat_template")
            or self.tokenizer.chat_template is None
        ):
            self.tokenizer.chat_template = self.DEFAULT_CHAT_TEMPLATE

    def get_raw_dataset(self, cache_dir: Optional[str] = None):
        """
        Load the raw dataset from Hugging Face, using cached copy if available.
        Additionally reformats the entries to fit the alpaca template.

        :param cache_dir: disk location to search for cached dataset
        :return: the requested dataset
        """
        raw_dataset = super().get_raw_dataset(cache_dir=cache_dir)

        # helper fn for restructuring each dataset entry using the chat template
        def restructure_fn(sample):
            if sample["messages"][0]["role"] != "system":
                sample["messages"].insert(0, {"role": "system", "content": ""})

            sample["messages"] = self.tokenizer.apply_chat_template(
                sample["messages"], tokenize=False, add_generation_prompt=False
            )
            return sample

        raw_dataset = self.map(
            raw_dataset,
            function=restructure_fn,
            batched=False,
            remove_columns=[],
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Restructuring Ultra Chat Dataset",
        )
        return raw_dataset
