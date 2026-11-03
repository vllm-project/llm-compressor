"""Calibration tokenization must emit a single leading BOS token.

Llama-style chat templates already render the BOS token, and the tokenizer
post-processor prepends another when ``add_special_tokens`` stays at its
default. Inference via ``apply_chat_template(tokenize=True)`` emits only one, so
calibration samples must not begin with a doubled BOS token. See
https://github.com/vllm-project/llm-compressor/issues/3284
"""

import pytest
from transformers import AutoProcessor

from llmcompressor.args import DatasetArguments
from llmcompressor.datasets import get_processed_dataset

# A Llama-3-style chat template that renders the BOS token itself, matching the
# behavior of real instruct tokenizers. Chat datasets honor an already-set
# chat_template, so this is what makes the double-BOS regression observable: the
# template emits BOS and, without the fix, the tokenizer prepends a second one.
BOS_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}"
)


def _leading_sequence(sample: dict) -> list[int]:
    """Return the token id sequence, unwrapping any extra multimodal dim."""
    ids = sample.get("input_ids", sample.get("decoder_input_ids"))
    if ids and not isinstance(ids[0], int):  # nested [[...]] from some processors
        ids = ids[0]
    return ids


@pytest.mark.unit
@pytest.mark.parametrize(
    "dataset_key, dataset_config, split, processor_id",
    [
        ("perfectblend", None, "train[:8]", None),
        ("open_platypus", None, "train[:8]", None),
        ("evolcodealpaca", None, "train[:8]", None),
        ("gsm8k", "main", "train[:8]", None),
        ("ultrachat_200k", None, "train_sft[:8]", None),
        ("wikitext", "wikitext-2-raw-v1", "train[:8]", None),
        # multimodal pre-baked datasets; only their text tokenization is
        # exercised, using a processor stub matching the expected processor type
        ("flickr", None, "test[:2]", "Qwen/Qwen3-VL-8B-Instruct"),
        ("peoples_speech", None, "test[:2]", "openai/whisper-large-v3"),
    ],
)
def test_prebaked_datasets_single_bos(
    tiny_llama_path, dataset_key, dataset_config, split, processor_id
):
    """
    Load, process, and tokenize a pre-baked dataset and assert that no sample
    begins with two BOS tokens. "custom" is excluded: it is not a pre-baked HF
    dataset (it needs a local path).

    A sample starts with a doubled BOS when its first two token ids both equal
    the tokenizer's ``bos_token_id``. Datasets whose tokenizer defines no BOS
    token (or whose template does not emit one) simply have nothing to double,
    so the assertion is vacuously satisfied.
    """
    processor = AutoProcessor.from_pretrained(
        processor_id if processor_id is not None else tiny_llama_path
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)

    # Give the text tokenizer a chat template that emits BOS, so chat datasets
    # reproduce the double-BOS condition this test guards against.
    if processor_id is None:
        tokenizer.chat_template = BOS_CHAT_TEMPLATE

    dataset_args = DatasetArguments(
        dataset=dataset_key,
        dataset_config_name=dataset_config,
        splits=split,
    )
    dataset = get_processed_dataset(dataset_args=dataset_args, processor=processor)

    assert dataset is not None
    assert len(dataset) > 0

    if bos_token_id is None:
        return

    for i in range(len(dataset)):
        ids = _leading_sequence(dataset[i])
        if len(ids) >= 2 and ids[0] == bos_token_id:
            assert ids[1] != bos_token_id, (
                f"{dataset_key} sample {i} begins with two BOS tokens "
                f"({bos_token_id}); calibration must emit a single leading BOS"
            )
