import pytest
from transformers import AutoProcessor

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

    with pytest.raises(ValueError):
        # fails due to no test split specified
        split_datasets = make_dataset_splits(
            datasets, do_train=True, do_eval=True, do_predict=True
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("meta-llama/Meta-Llama-3-8B-Instruct", ["input_ids", "attention_mask"]),
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", ["input_ids", "attention_mask"]),
        (
            "Qwen/Qwen2-VL-2B-Instruct",
            [
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
            ],
        ),
        ("mgoin/pixtral-12b", ["input_ids", "attention_mask", "pixel_values"]),
        ("openai/whisper-large-v2", ["input_features"]),
        (
            "Qwen/Qwen2-Audio-7B-Instruct",
            ["input_ids", "attention_mask", "input_features", "feature_attention_mask"],
        ),
    ],
)
def test_processor_model_input_names(model_id, expected):
    """
    Tests the model_input_names attribute of common model processors
    """

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    assert processor.model_input_names == expected
