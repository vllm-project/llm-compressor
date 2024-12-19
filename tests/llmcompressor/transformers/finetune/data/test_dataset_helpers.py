from unittest.mock import Mock

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


@pytest.mark.unit
def test_datasets_fallbacks():
    # strict splits
    mock_datasets = {"calibration": Mock(ds_name="calibration_ds", column_names=[])}
    with pytest.raises(ValueError):
        _ = make_dataset_splits(mock_datasets, do_train=True)
    with pytest.raises(ValueError):
        _ = make_dataset_splits(mock_datasets, do_eval=True)
    with pytest.raises(ValueError):
        _ = make_dataset_splits(mock_datasets, do_predict=True)

    # validation, predict, and oneshot fallbacks
    mock_datasets = {"test": Mock(ds_name="test_ds", column_names=[])}
    with pytest.warns(UserWarning):
        split_ds = make_dataset_splits(
            mock_datasets, do_eval=True, do_predict=True, do_oneshot=True
        )
    assert split_ds.get("validation").ds_name == "test_ds"
    assert split_ds.get("test").ds_name == "test_ds"
    assert split_ds.get("calibration").ds_name == "test_ds"

    # oneshot takes train without warning
    mock_datasets = {"train": Mock(ds_name="train_ds", column_names=[])}
    split_ds = make_dataset_splits(mock_datasets, do_oneshot=True)
    assert split_ds.get("calibration").ds_name == "train_ds"

    # oneshot takes test with warning
    mock_datasets = {"test": Mock(ds_name="test_ds", column_names=[])}
    with pytest.warns(UserWarning):
        split_ds = make_dataset_splits(mock_datasets, do_oneshot=True)
    assert split_ds.get("calibration").ds_name == "test_ds"

    # oneshot takes custom splits with warning
    mock_datasets = {"custom_split": Mock(ds_name="custom_ds", column_names=[])}
    with pytest.warns(UserWarning):
        split_ds = make_dataset_splits(mock_datasets, do_oneshot=True)
    assert split_ds.get("calibration").ds_name == "custom_ds"
