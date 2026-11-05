import json
from copy import deepcopy
from typing import TYPE_CHECKING

from loguru import logger

from llmcompressor.transformers.data import TextGenerationDataset
from llmcompressor.typing import Processor

if TYPE_CHECKING:
    from llmcompressor.args import DatasetArguments

# optional chat message fields which are null or empty for part of the corpus;
# they are dropped during preprocessing since chat templates may reject them
_EMPTY_MESSAGE_KEYS = ("tool_calls", "tool_call_id", "name", "reasoning_content")


def _parse_json_field(value):
    """
    Parse a JSON string recorded in the corpus, returning the original value
    unchanged if it is not a string or does not decode

    :param value: field value, recorded as a JSON string
    :return: decoded value, or the original value if it cannot be decoded
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _clean_messages(messages: list[dict]) -> list[dict]:
    """
    Normalize the messages recorded in a RazorCal sample for chat templates:
    drop optional fields which are null or empty, and decode tool call
    arguments, which are recorded as JSON strings, back into objects

    :param messages: messages recorded in a RazorCal sample
    :return: normalized messages
    """
    cleaned = []
    for message in messages:
        message = {
            key: value
            for key, value in message.items()
            if key not in _EMPTY_MESSAGE_KEYS or value
        }
        if message.get("tool_calls"):
            message["tool_calls"] = [
                _clean_tool_call(tool_call) for tool_call in message["tool_calls"]
            ]
        cleaned.append(message)
    return cleaned


def _clean_tool_call(tool_call: dict) -> dict:
    """
    Decode the arguments of a recorded tool call back into an object

    :param tool_call: tool call recorded in a RazorCal message
    :return: tool call with decoded arguments
    """
    function = dict(tool_call.get("function") or {})
    function["arguments"] = _parse_json_field(function.get("arguments"))
    return {**tool_call, "function": function}


def _clean_tools(tools: list[dict] | None) -> list[dict] | None:
    """
    Normalize the tool definitions recorded alongside a RazorCal sample,
    decoding their JSON schema parameters back into objects

    :param tools: tool definitions recorded in a RazorCal sample
    :return: normalized tool definitions, or None if none are recorded
    """
    if not tools:
        return None

    cleaned = []
    for tool in tools:
        function = dict(tool.get("function") or {})
        function["parameters"] = _parse_json_field(function.get("parameters"))
        cleaned.append({**tool, "function": function})
    return cleaned


@TextGenerationDataset.register(name="razorcal")
class RazorCalDataset(TextGenerationDataset):
    """
    Child text generation class for the RazorCal calibration corpus

    RazorCal is the multi-domain calibration corpus released with the RAZOR
    expert pruning method (https://github.com/nick7nlp/RAZOR). It contains
    2,048 chat samples across seven domains: 512 Coding samples and 256 each
    of Math, Science (STEM), Chinese-STEM, Instruction Following, Tool Calling
    and World Knowledge. Samples are rendered with the processor's chat
    template, forwarding any tool definitions recorded alongside the sample;
    samples which fail to render fall back to their raw message contents.

    The corpus is published
    (https://huggingface.co/datasets/inference-optimization/RazorCal) with one
    row per sample, keeping the `domain` and `sub_source` attribution columns.
    Tool call arguments and tool definition parameters are recorded as JSON
    strings and are decoded back into objects before rendering. Only the
    `train` split exists; use `num_calibration_samples` to control how many
    of the samples are used for calibration.

    :param dataset_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `train` or
        `train[:512]`. Only the `train` split is available
    :param processor: processor or tokenizer to use on dataset
    """

    add_special_tokens = False

    # like the default chat template of the other prebaked chat datasets, but
    # with a branch for the tool messages recorded in the Tool Calling domain
    DEFAULT_CHAT_TEMPLATE = (
        "{% for message in messages %}\n"
        "{% if message['role'] == 'user' %}\n"
        "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
        "{% elif message['role'] == 'system' %}\n"
        "{{ '<|system|>\n' + message['content'] + eos_token }}\n"
        "{% elif message['role'] == 'assistant' %}\n"
        "{{ '<|assistant|>\n' + message['content'] + eos_token }}\n"
        "{% elif message['role'] == 'tool' %}\n"
        "{{ '<|tool|>\n' + message['content'] + eos_token }}\n"
        "{% endif %}\n"
        "{% if loop.last and add_generation_prompt %}\n"
        "{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    )

    def __init__(
        self, dataset_args: "DatasetArguments", split: str, processor: Processor
    ):
        dataset_args = deepcopy(dataset_args)
        dataset_args.dataset = "inference-optimization/RazorCal"
        dataset_args.text_column = "messages"

        super().__init__(dataset_args=dataset_args, split=split, processor=processor)

        self._template_fallback_logged = False

        if (
            self.tokenizer is not None
            and getattr(self.tokenizer, "chat_template", None) is None
        ):
            self.tokenizer.chat_template = self.DEFAULT_CHAT_TEMPLATE
            logger.warning(
                "tokenizer.chat_template is not set, using default chat template for "
                f"{self.__class__.__name__}"
            )

    def dataset_template(self, sample):
        messages = _clean_messages(sample["messages"])
        tools = _clean_tools(sample.get("tools"))

        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                tools=tools,
            )
        except Exception as error:
            if not self._template_fallback_logged:
                self._template_fallback_logged = True
                logger.warning(
                    "RazorCal samples could not be rendered with the processor's "
                    f"chat template ({error}); falling back to the raw message "
                    "contents of affected samples"
                )
            text = "\n".join(message.get("content", "") for message in messages)

        return {"text": text}
