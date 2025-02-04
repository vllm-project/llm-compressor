import csv
import json
import os
import shutil
from functools import wraps

import pytest
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from llmcompressor.datasets import get_raw_dataset
from llmcompressor.transformers.finetune.data import DataTrainingArguments

CACHE_DIR = "/tmp/cache_dir"


def create_mock_dataset_files(tmp_dir, file_extension):
    train_entries = [
        {"id": 1, "question": "What is 2 + 2?", "answer": "4"},
        {"id": 2, "question": "What is the capital of France?", "answer": "Paris"},
        {"id": 3, "question": "Who wrote '1984'?", "answer": "George Orwell"},
        {"id": 4, "question": "What is the largest planet?", "answer": "Jupiter"},
        {"id": 5, "question": "What is the boiling point of water?", "answer": "100°C"},
    ]

    test_entries = [
        {"id": 6, "question": "What is 3 + 5?", "answer": "8"},
        {"id": 7, "question": "What is the capital of Germany?", "answer": "Berlin"},
        {"id": 8, "question": "Who wrote 'The Hobbit'?", "answer": "J.R.R. Tolkien"},
        {
            "id": 9,
            "question": "What planet is known as the Red Planet?",
            "answer": "Mars",
        },
        {"id": 10, "question": "What is the freezing point of water?", "answer": "0°C"},
    ]

    train_file_path = os.path.join(tmp_dir, f"train.{file_extension}")
    test_file_path = os.path.join(tmp_dir, f"test.{file_extension}")
    os.makedirs(tmp_dir, exist_ok=True)

    def _write_file(entries, file_path):
        if file_extension == "json":
            with open(file_path, "w") as json_file:
                for entry in entries:
                    json_file.write(json.dumps(entry) + "\n")
        elif file_extension == "csv":
            fieldnames = ["id", "question", "answer"]
            with open(file_path, "w", newline="") as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                csv_writer.writeheader()
                csv_writer.writerows(entries)

    _write_file(train_entries, train_file_path)
    _write_file(test_entries, test_file_path)


@pytest.fixture
def data_arguments_fixture():
    @wraps(DataTrainingArguments)
    def get_data_args(**dataset_kwargs):
        return DataTrainingArguments(**dataset_kwargs)

    return get_data_args


@pytest.mark.parametrize(
    "dataset_kwargs",
    [
        (
            {
                "dataset": "HuggingFaceH4/ultrachat_200k",
                "load_dataset_args": {
                    "split": "train_sft",
                },
            }
        ),
        ({"dataset": "openai/gsm8k", "load_dataset_args": {"name": "main"}}),
    ],
)
def test_load_dataset__hf_dataset_alias(data_arguments_fixture, dataset_kwargs):
    dataset_path_name = os.path.join(
        CACHE_DIR,
        dataset_kwargs["dataset"].split("/")[-1],
    )
    dataset_kwargs["load_dataset_args"]["cache_dir"] = dataset_path_name

    data_args = data_arguments_fixture(**dataset_kwargs)
    dataset = get_raw_dataset(data_args.dataset, **data_args.load_dataset_args)

    assert isinstance(
        dataset, (Dataset, DatasetDict, IterableDataset, IterableDatasetDict)
    )


def test_load_dataset__hf_dataset_path(data_arguments_fixture):
    dataset_folders = [
        name
        for name in os.listdir(CACHE_DIR)
        if os.path.isdir(os.path.join(CACHE_DIR, name))
    ]

    for dataset_folder in dataset_folders:
        dataset_path = os.path.join(CACHE_DIR, dataset_folder)
        dataset_kwargs = {"dataset": dataset_path}

        data_args = data_arguments_fixture(**dataset_kwargs)

        try:
            dataset = get_raw_dataset(data_args.dataset, **data_args.load_dataset_args)
            assert isinstance(
                dataset, (Dataset, DatasetDict, IterableDataset, IterableDatasetDict)
            )
        finally:
            shutil.rmtree(dataset_path)


@pytest.mark.parametrize("file_extension", ["json", "csv"])
def test_load_dataset__local_dataset_path(file_extension, data_arguments_fixture):
    dataset_path = os.path.join(CACHE_DIR, "mock_dataset")
    create_mock_dataset_files(dataset_path, file_extension)

    try:
        dataset = get_raw_dataset(dataset_path)

        assert isinstance(dataset, (Dataset, DatasetDict))
        assert "train" in dataset and "test" in dataset
        assert len(dataset["train"]) == 5
        assert len(dataset["test"]) == 5

    finally:
        shutil.rmtree(dataset_path)
