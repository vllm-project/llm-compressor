from copy import deepcopy
from typing import TYPE_CHECKING

from loguru import logger

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.typing import Processor

if TYPE_CHECKING:
    from llmcompressor.args import DatasetArguments


@TextGenerationDataset.register(name="ultrachat_200k")
class UltraChatDataset(TextGenerationDataset):
    """
    Child text generation class for the Ultra Chat 200k dataset

    :param dataset_args: configuration settings for dataset loading
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

    def __init__(
        self, dataset_args: "DatasetArguments", split: str, processor: Processor
    ):
        dataset_args = deepcopy(dataset_args)
        dataset_args.dataset = "HuggingFaceH4/ultrachat_200k"
        dataset_args.text_column = "messages"

        if split in ["train", "test"]:
            split += "_sft"

        super().__init__(dataset_args=dataset_args, split=split, processor=processor)

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
