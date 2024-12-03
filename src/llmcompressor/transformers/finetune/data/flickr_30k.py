from copy import deepcopy
from typing import TYPE_CHECKING

from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.utils import Processor

if TYPE_CHECKING:
    from llmcompressor.transformers import DataTrainingArguments as DataArgs


@TextGenerationDataset.register(name="flickr", alias="flickr30k")
class Flickr30K(TextGenerationDataset):
    """
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
        data_args.dataset = "lmms-lab/flickr30k"

        super().__init__(data_args=data_args, split=split, processor=processor)

        if (
            self.tokenizer is not None
            and getattr(self.tokenizer, "chat_template", None) is None
        ):
            # note that since tokenizer is a member of processor,
            # this change affects processor.apply_chat_template
            self.tokenizer.chat_template = self.DEFAULT_CHAT_TEMPLATE

    def dataset_template(self, sample):
        if self.processor is None:
            raise ValueError("TODO")

        breakpoint()

        messages = sample["messages"]
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})

        return {
            "text": self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
            "image": sample["image"],
        }
