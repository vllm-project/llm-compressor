import csv
import json
import os
import tempfile
from io import StringIO
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from llmcompressor import train
from tests.testing_utils import parse_params, requires_gpu

CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/finetune/finetune_custom"
GPU_CONFIGS_DIRECTORY = "tests/llmcompressor/transformers/finetune/finetune_custom/gpu"


def create_mock_file(extension, content, path, filename):
    os.makedirs(path, exist_ok=True)

    if extension == "json":
        mock_data = {"text": content}
        mock_content = json.dumps(mock_data, indent=2)

    else:
        fieldnames = ["text"]
        mock_data = [{"text": content}]
        csv_output = StringIO()
        csv_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(mock_data)
        mock_content = csv_output.getvalue()

    mock_filename = f"{filename}.{extension}"
    mock_filepath = os.path.join(path, mock_filename)

    with open(mock_filepath, "w") as mock_file:
        mock_file.write(mock_content)

    return mock_filepath  # Return the file path


def create_mock_custom_dataset_folder_structure(tmp_dir_data, file_extension):
    train_path = os.path.join(tmp_dir_data, "train")
    test_path = os.path.join(tmp_dir_data, "test")
    validate_path = os.path.join(tmp_dir_data, "validate")

    # create tmp mock data files
    create_mock_file(
        extension=file_extension,
        content="text for train data 1",
        path=train_path,
        filename="data1",
    )
    create_mock_file(
        extension=file_extension,
        content="text for train data 2",
        path=train_path,
        filename="data2",
    )
    create_mock_file(
        extension=file_extension,
        content="text for test data 1",
        path=test_path,
        filename="data3",
    )
    create_mock_file(
        extension=file_extension,
        content="text for validate data 1",
        path=validate_path,
        filename="data4",
    )
    return True


def _test_finetune_wout_recipe_custom_dataset(
    model, file_extension, num_train_epochs, output
):
    dataset_path = Path(tempfile.mkdtemp())

    created_success = create_mock_custom_dataset_folder_structure(
        dataset_path, file_extension
    )
    assert created_success

    def preprocessing_func(example):
        example["text"] = "Review: " + example["text"]
        return example

    concatenate_data = False

    train(
        model=model,
        dataset=file_extension,
        output_dir=output,
        recipe=None,
        num_train_epochs=num_train_epochs,
        concatenate_data=concatenate_data,
        text_column="text",
        dataset_path=dataset_path,
        preprocessing_func=preprocessing_func,
        precision="bfloat16",
        bf16=True,
    )


@pytest.mark.integration
@pytest.mark.parametrize("config", parse_params(CONFIGS_DIRECTORY))
def test_oneshot_then_finetune_small(config, tmp_path):
    model = config["model"]
    file_extension = config["file_extension"]
    num_train_epochs = config["num_train_epochs"]

    output = tmp_path / "oneshot_output"

    _test_finetune_wout_recipe_custom_dataset(
        model, file_extension, num_train_epochs, output
    )


@requires_gpu
@pytest.mark.integration
@pytest.mark.parametrize("config", parse_params(GPU_CONFIGS_DIRECTORY))
def test_oneshot_then_finetune_gpu(config, tmp_path):
    model = config["model"]
    file_extension = config["file_extension"]
    num_train_epochs = config["num_train_epochs"]
    output = tmp_path / "oneshot_output"

    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        model, device_map=device, torch_dtype=torch.bfloat16
    )
    _test_finetune_wout_recipe_custom_dataset(
        model, file_extension, num_train_epochs, output
    )
