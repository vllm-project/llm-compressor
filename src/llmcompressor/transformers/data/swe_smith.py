import json
from copy import deepcopy
from typing import TYPE_CHECKING

from loguru import logger

from llmcompressor.transformers.data import TextGenerationDataset
from llmcompressor.typing import Processor

if TYPE_CHECKING:
    from llmcompressor.args import DatasetArguments


@TextGenerationDataset.register(name="swe_smith")
class SWESmithDataset(TextGenerationDataset):
    """
    Child text generation class for the SWE-smith agent trajectories dataset. Use
    the `tool` split (for instance `tool[:512]`) for OpenAI-style tool calling.

    Each trajectory is converted to standard chat messages so that tool calling is
    rendered by the model's chat template:
    * list-of-parts content -> plain string content
    * assistant `tool_calls` -> `tool_calls` with parsed (dict) arguments
    * `tool` observations `tool_call_ids` -> `tool_call_id`
    * dataset-specific fields (`agent`, `message_type`, `action`, ...) are dropped.
      Assistant `thought` is dropped rather than mapped to `reasoning_content`, since
      it always duplicates the assistant `content`

    :param dataset_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `tool` or `tool[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    def __init__(
        self, dataset_args: "DatasetArguments", split: str, processor: Processor
    ):
        dataset_args = deepcopy(dataset_args)
        dataset_args.dataset = "SWE-bench/SWE-smith-trajectories"
        dataset_args.text_column = "messages"

        super().__init__(dataset_args=dataset_args, split=split, processor=processor)

        if (
            self.tokenizer is not None
            and getattr(self.tokenizer, "chat_template", None) is None
        ):
            logger.warning(
                "tokenizer.chat_template is not set, using default chat template for "
                f"{self.__class__.__name__}"
            )

    def dataset_template(self, sample):
        messages = [_to_chat_message(msg) for msg in json.loads(sample["messages"])]
        return {
            "text": self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        }


def _to_chat_message(message: dict) -> dict:
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part["text"] for part in content if part["type"] == "text")

    chat_message = {"role": message["role"], "content": content}

    if message.get("tool_calls"):
        chat_message["tool_calls"] = [
            {
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["function"]["name"],
                    "arguments": json.loads(tool_call["function"]["arguments"]),
                },
            }
            for tool_call in message["tool_calls"]
        ]

    if message.get("tool_call_ids"):
        chat_message["tool_call_id"] = message["tool_call_ids"][0]

    return chat_message
