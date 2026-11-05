import json
import re
from copy import deepcopy
from typing import TYPE_CHECKING

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from loguru import logger

from llmcompressor.transformers.data import TextGenerationDataset
from llmcompressor.typing import Processor

if TYPE_CHECKING:
    from llmcompressor.args import DatasetArguments

# optional chat message fields which are null or empty for part of the corpus;
# they are dropped during preprocessing since chat templates may reject them
_EMPTY_MESSAGE_KEYS = ("tool_calls", "tool_call_id", "name", "reasoning_content")

# `split[from:to]` slicing instructions supported by `datasets.load_dataset`,
# where each bound is an absolute index, a percentage suffixed with `%`, or empty
_SPLIT_SPEC_RE = re.compile(
    r"^(?P<split>\w+)\["
    r"(?P<from>-?\d*)(?P<from_pct>%?)"
    r":"
    r"(?P<to>-?\d*)(?P<to_pct>%?)"
    r"\]$"
)


def _resolve_bound(bound: str, percent: bool, num_samples: int, default: int) -> int:
    """
    Resolve a single slicing bound to an absolute sample index, following the
    resolution of `datasets.load_dataset` slicing instructions: percentages are
    rounded to the closest index, negative bounds count back from the end of
    the dataset, and bounds are clamped to the number of samples

    :param bound: bound captured by `_SPLIT_SPEC_RE`, possibly empty
    :param percent: whether the bound is a percentage of the dataset
    :param num_samples: total number of samples in the dataset
    :param default: index used when the bound is empty
    :return: resolved absolute sample index
    """
    if bound == "":
        return default

    index = int(bound)
    if percent:
        if abs(index) > 100:
            raise ValueError(
                f"Percent slice boundary {bound} is out of range [-100, 100]"
            )
        index = round(index * num_samples / 100)
    if index < 0:
        index = max(num_samples + index, 0)
    return min(index, num_samples)


def _split_spec_range(spec: "re.Match[str]", num_samples: int) -> range:
    """
    Resolve a `split[from:to]` slicing instruction to a range of sample indices

    :param spec: match of `_SPLIT_SPEC_RE` against the split specification
    :param num_samples: total number of samples in the dataset
    :return: range of sample indices selected by the specification
    """
    start = _resolve_bound(
        spec.group("from"), bool(spec.group("from_pct")), num_samples, default=0
    )
    stop = _resolve_bound(
        spec.group("to"),
        bool(spec.group("to_pct")),
        num_samples,
        default=num_samples,
    )
    if start >= stop:
        raise ValueError(f"Instruction {spec.group(0)!r} corresponds to no data!")
    return range(start, stop)


def _clean_messages(messages: list[dict]) -> list[dict]:
    """
    Drop optional message fields which are null or empty in part of the corpus,
    since chat templates may reject or misrender them

    :param messages: messages recorded in a RazorCal sample
    :return: messages with empty optional fields removed
    """
    return [
        {
            key: value
            for key, value in message.items()
            if key not in _EMPTY_MESSAGE_KEYS or value
        }
        for message in messages
    ]


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

    The corpus is published (https://huggingface.co/datasets/Nickyang/RazorCal)
    as a single JSON object holding a `meta` summary and the full sample list
    under `data`. Samples are flattened out of that object, keeping the `domain`
    and `sub_source` attribution columns and carrying the `messages` and `tools`
    chat payloads as JSON strings, since their free-form contents cannot be
    reconciled into a single arrow schema. Slicing instructions such as
    `train[:512]` are resolved against the flattened samples for the same
    reason, and only the `train` split exists. Use `num_calibration_samples`
    to control how many of the samples are used for calibration.

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
        dataset_args.dataset = "Nickyang/RazorCal"
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

    def load_dataset(self):
        # load the named split only; slicing bounds are applied to the
        # flattened samples below, since slicing the raw single-row dataset
        # first would clamp every bound to a single sample
        spec = None
        split = self.split
        if isinstance(split, str) and "[" in split:
            spec = _SPLIT_SPEC_RE.match(split)
            if spec is None:
                raise ValueError(
                    f"Split specification {split!r} is not supported by the razorcal "
                    "dataset. Use the 'train' split, optionally with absolute or "
                    "percentage bounds (for example 'train[:512]'), or control the "
                    "number of samples with `num_calibration_samples`"
                )
            self.split = spec.group("split")

        try:
            raw_dataset = super().load_dataset()
        finally:
            self.split = split

        if isinstance(raw_dataset, (DatasetDict, IterableDatasetDict)):
            key = "train" if "train" in raw_dataset else next(iter(raw_dataset))
            raw_dataset = raw_dataset[key]
        if isinstance(raw_dataset, IterableDataset):
            # the corpus is a single JSON object, so streaming still has to
            # resolve the full sample list
            row = next(iter(raw_dataset))
        else:
            row = raw_dataset[0]

        if "data" not in row:
            raise ValueError(
                "Expected the Nickyang/RazorCal corpus to store its samples under a "
                f"single 'data' column, found {sorted(row.keys())}. The razorcal "
                "dataset class must be updated to match the published layout"
            )

        # tool call arguments and tool definitions are free-form JSON whose
        # value types conflict across records (for example a string and a
        # number under the same key path of different records), which arrow
        # cannot reconcile into a single schema; keep the attribution columns
        # flat and carry the chat payloads as JSON strings instead
        dataset = Dataset.from_list(
            [
                {
                    "domain": record.get("domain"),
                    "sub_source": record.get("sub_source"),
                    "coding_subclass": record.get("coding_subclass"),
                    "messages": json.dumps(record["messages"], ensure_ascii=False),
                    "tools": json.dumps(record.get("tools") or [], ensure_ascii=False),
                }
                for record in row["data"]
            ]
        )

        if spec is not None:
            dataset = dataset.select(_split_spec_range(spec, len(dataset)))

        return dataset

    def dataset_template(self, sample):
        messages = _clean_messages(json.loads(sample["messages"]))
        tools = json.loads(sample["tools"]) or None

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
