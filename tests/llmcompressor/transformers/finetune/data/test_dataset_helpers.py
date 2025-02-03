import pytest

from llmcompressor.transformers.finetune.data.data_args import DataTrainingArguments
from llmcompressor.transformers.finetune.data.data_helpers import (
    get_raw_dataset,
    make_dataset_splits,
)


@pytest.mark.unit
def test_combined_datasets():
    data_args = DataTrainingArguments(
        dataset="wikitext", dataset_config_name="wikitext-2-raw-v1"
    )
    raw_wikitext2 = get_raw_dataset(
        data_args.dataset,
        name=data_args.dataset_config_name,
        splits=data_args.splits,
        streaming=data_args.streaming,
    )
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
    for split_name, _ in splits.items():
        raw_wikitext2 = get_raw_dataset(
            data_args.dataset,
            name=data_args.dataset_config_name,
            splits=data_args.splits,
            streaming=data_args.streaming,
        )
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
