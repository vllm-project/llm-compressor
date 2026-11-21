from types import SimpleNamespace

import pytest
from compressed_tensors.modeling import attention as ct_attention
from datasets import Dataset
from loguru import logger
from torch.utils.data import DataLoader, SequentialSampler

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.datasets.utils import (
    DataCollatorWithFlattening,
    TokenBudgetBatchSampler,
    batch_by_tokens,
    check_flattening_support,
    format_calibration_data,
    get_attention_implementation,
)


@pytest.mark.unit
def test_flattening_packs_samples_into_one_row():
    batch = DataCollatorWithFlattening()(
        [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}, {"input_ids": [4, 5]}]
    )
    assert batch.keys() == {"input_ids", "position_ids"}
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1]]


@pytest.mark.unit
def test_flattening_keeps_loss_mask_and_drops_masked_tokens():
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 0], "loss_mask": [0, 1, 1]},
        {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1], "loss_mask": [1, 0, 1]},
    ]
    batch = DataCollatorWithFlattening(max_seq_length=2)(features)
    assert batch["input_ids"].tolist() == [[1, 2, 4, 5]]
    assert batch["position_ids"].tolist() == [[0, 1, 0, 1]]
    assert batch["loss_mask"].tolist() == [[0, 1, 1, 0]]


@pytest.mark.unit
def test_flattening_refuses_non_text_samples():
    with pytest.raises(ValueError, match="pixel_values"):
        DataCollatorWithFlattening()([{"input_ids": [1], "pixel_values": [0.0]}])


@pytest.mark.unit
def test_flatten_data_collator_argument():
    args = DatasetArguments(data_collator="flatten", max_seq_length=8)
    dataset = Dataset.from_dict({"input_ids": [[1, 2], [3]]})
    dataloader = format_calibration_data(args, dataset, processor=None)
    assert isinstance(dataloader.collate_fn, DataCollatorWithFlattening)
    assert dataloader.collate_fn.max_seq_length == 8


@pytest.mark.unit
def test_packed_batches_fill_budget_in_sampler_order():
    lengths = [5, 2, 9, 3, 4, 20]
    sampler = TokenBudgetBatchSampler(
        [1, 0, 3, 4, 2, 5], lengths, max_tokens=10, packed=True
    )
    # sums 2+5+3, 4, 9, then 20 alone since it exceeds the budget
    assert sampler.batches == [[1, 0, 3], [4], [2], [5]]


@pytest.mark.unit
def test_batch_by_tokens_packs_with_flattening_collator():
    dataset = Dataset.from_dict({"input_ids": [[1] * n for n in (6, 3, 3, 2)]})
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        collate_fn=DataCollatorWithFlattening(),
    )
    rebuilt = batch_by_tokens(dataloader, max_tokens=9)
    shapes = [tuple(batch["input_ids"].shape) for batch in rebuilt]
    assert shapes == [(1, 9), (1, 5)]


def _model(attn_implementation="flex_attention", model_type="llama"):
    return SimpleNamespace(
        config=SimpleNamespace(
            _attn_implementation=attn_implementation, model_type=model_type
        )
    )


def _warnings(function, *args):
    messages = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        function(*args)
    finally:
        logger.remove(sink)
    return messages


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_type", ["falcon", "bloom", "mpt", "opt", "qwen3_next", "kimi_linear", None]
)
def test_flattening_refuses_models_not_known_to_keep_samples_apart(model_type):
    with pytest.raises(ValueError, match="not known to"):
        check_flattening_support(_model(model_type=model_type))


@pytest.mark.unit
def test_flattening_reads_the_text_config():
    text_config = SimpleNamespace(model_type="qwen3")
    config = SimpleNamespace(
        _attn_implementation="flex_attention", get_text_config=lambda: text_config
    )
    check_flattening_support(SimpleNamespace(config=config))


@pytest.mark.unit
@pytest.mark.parametrize(
    "attn_implementation,warns",
    [
        ("sdpa", True),
        ("eager", True),
        ("flex_attention", False),
        ("flash_attention_2", False),
        ("kernels-community/flash-attn3", False),
    ],
)
def test_flattening_warns_about_full_attention_masks(attn_implementation, warns):
    messages = _warnings(check_flattening_support, _model(attn_implementation))
    assert any("attention mask" in message for message in messages) == warns


@pytest.mark.unit
@pytest.mark.parametrize("original,warns", [("flex_attention", False), ("sdpa", True)])
def test_flattening_looks_through_hooked_attention(monkeypatch, original, warns):
    # attention and KV cache quantization replace the configured implementation
    monkeypatch.setattr(ct_attention.QuantizedAttentionImpl, "_original_impl", original)
    model = _model(ct_attention.HOOKED_ATTENTION_NAME)
    assert get_attention_implementation(model) == original

    messages = _warnings(check_flattening_support, model)
    assert any("attention mask" in message for message in messages) == warns


@pytest.mark.unit
def test_flattening_refuses_partial_loss_masks():
    features = [{"input_ids": [1, 2], "loss_mask": [1, 1]}, {"input_ids": [3]}]
    with pytest.raises(ValueError, match="loss_mask"):
        DataCollatorWithFlattening()(features)


@pytest.mark.unit
def test_flattening_drops_single_segment_token_type_ids():
    batch = DataCollatorWithFlattening()(
        [{"input_ids": [1, 2], "token_type_ids": [0, 0]}, {"input_ids": [3]}]
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3]]

    with pytest.raises(ValueError, match="token_type_ids"):
        DataCollatorWithFlattening()([{"input_ids": [1, 2], "token_type_ids": [0, 1]}])


@pytest.mark.unit
@pytest.mark.parametrize(
    "data_collator,warns", [("flatten", False), ("truncation", True)]
)
def test_shuffle_warning_only_for_padding_or_truncation(data_collator, warns):
    args = DatasetArguments(batch_size=4, data_collator=data_collator)
    dataset = Dataset.from_dict({"input_ids": [[1, 2], [3]]})
    messages = _warnings(format_calibration_data, args, dataset, None)
    assert any("Shuffling" in message for message in messages) == warns
