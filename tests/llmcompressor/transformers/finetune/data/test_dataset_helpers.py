# TODO: rename to `test_data_helpers.py`
import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from llmcompressor.transformers.finetune.data.data_args import DataTrainingArguments
from llmcompressor.transformers.finetune.data.data_helpers import (
    format_calibration_data,
    get_raw_dataset,
    make_dataset_splits,
)
from llmcompressor.transformers.finetune.text_generation import configure_processor


@pytest.mark.unit
def test_combined_datasets():
    data_args = DataTrainingArguments(
        dataset="wikitext", dataset_config_name="wikitext-2-raw-v1"
    )
    raw_wikitext2 = get_raw_dataset(data_args)
    datasets = {"all": raw_wikitext2}

    split_datasets = make_dataset_splits(
        datasets, do_train=True, do_eval=True, do_predict=True
    )
    assert split_datasets.get("train") is not None
    assert split_datasets.get("validation") is not None
    assert split_datasets.get("test") is not None

    split_datasets = make_dataset_splits(
        datasets, do_train=True, do_eval=False, do_predict=True
    )
    assert split_datasets.get("train") is not None
    assert split_datasets.get("validation") is None
    assert split_datasets.get("test") is not None


@pytest.mark.unit
def test_separate_datasets():
    splits = {"train": "train[:10%]", "validation": "train[10%:20%]"}
    data_args = DataTrainingArguments(
        dataset="wikitext", dataset_config_name="wikitext-2-raw-v1"
    )
    datasets = {}
    for split_name, split_str in splits.items():
        raw_wikitext2 = get_raw_dataset(data_args, split=split_str)
        datasets[split_name] = raw_wikitext2

    split_datasets = make_dataset_splits(
        datasets, do_train=True, do_eval=True, do_predict=False
    )
    assert split_datasets.get("train") is not None
    assert split_datasets.get("validation") is not None
    assert split_datasets.get("test") is None

    with pytest.raises(ValueError):
        # fails due to no test split specified
        split_datasets = make_dataset_splits(
            datasets, do_train=True, do_eval=True, do_predict=True
        )


@pytest.mark.unit
def test_format_calibration_data_padded_tokenized():
    vocab_size = 512
    seq_len = 2048
    ds_size = 16
    padded_tokenized_dataset = Dataset.from_dict(
        {"input_ids": torch.randint(0, vocab_size, (ds_size, seq_len))}
    )

    calibration_dataloader = format_calibration_data(
        padded_tokenized_dataset, num_calibration_samples=8, batch_size=4
    )

    batch = next(iter(calibration_dataloader))
    assert batch["input_ids"].size(0) == 4


@pytest.mark.unit
def test_format_calibration_data_unpaddded_tokenized():
    vocab_size = 512
    ds_size = 16
    unpadded_tokenized_dataset = Dataset.from_dict(
        {
            "input_ids": [
                torch.randint(0, vocab_size, (seq_len,)) for seq_len in range(ds_size)
            ]
        }
    )
    processor = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    configure_processor(processor)

    calibration_dataloader = format_calibration_data(
        unpadded_tokenized_dataset,
        num_calibration_samples=8,
        batch_size=4,
        processor=processor,
    )

    batch = next(iter(calibration_dataloader))
    assert batch["input_ids"].size(0) == 2
