import pytest

from llmcompressor.args import DatasetArguments
from llmcompressor.datasets import make_dataset_splits
from llmcompressor.transformers.finetune.data.data_helpers import get_raw_dataset


@pytest.mark.unit
def test_combined_datasets():
    dataset_args = DatasetArguments(
        dataset="wikitext", dataset_config_name="wikitext-2-raw-v1"
    )
    raw_wikitext2 = get_raw_dataset(dataset_args)
    datasets = {"all": raw_wikitext2}
    split_datasets = make_dataset_splits(datasets, do_train=True)
    assert split_datasets.get("train") is not None

    split_datasets = make_dataset_splits(datasets, do_train=True)
    assert split_datasets.get("train") is not None


@pytest.mark.unit
def test_separate_datasets():
    splits = {"train": "train[:5%]", "validation": "train[10%:20%]"}
    dataset_args = DatasetArguments(
        dataset="wikitext", dataset_config_name="wikitext-2-raw-v1"
    )
    datasets = {}
    for split_name, split_str in splits.items():
        raw_wikitext2 = get_raw_dataset(dataset_args, split=split_str)
        datasets[split_name] = raw_wikitext2

    split_datasets = make_dataset_splits(datasets, do_train=True)
    assert split_datasets.get("train") is not None

    with pytest.raises(ValueError):
        # fails due to no test split specified

        datasets.pop("train")
        split_datasets = make_dataset_splits(datasets, do_train=True)
