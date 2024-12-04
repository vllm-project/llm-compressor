import logging
import os
import warnings
from typing import Any, Callable, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.data import default_data_collator

from llmcompressor.typing import DatasetType

LOGGER = logging.getLogger(__name__)
LABELS_MASK_VALUE = -100

__all__ = [
    "create_batch_dataloader",
    "format_calibration_data",
    "get_raw_dataset",
    "make_dataset_splits",
    "get_custom_datasets_from_path",
    "LABELS_MASK_VALUE",
]


def create_batch_dataloader(
    dataloader: torch.utils.data.DataLoader,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader whose batch size is equal to the size of the dataset

    :param dataset: dataset used to generate dataloader
    :param batch_size: batch size of new dataloader
    :return: dataloader
    """
    dataset = dataloader.dataset
    sampler = dataloader.sampler.__class__(dataset)

    def pad_sequences(batch):
        # extract input_ids and attention_mask from the batch
        input_ids = [torch.tensor(item["input_ids"]).squeeze(0) for item in batch]
        masks = [torch.tensor(item["attention_mask"]).squeeze(0) for item in batch]

        # while 0 is not necessarily the "correct" padding value, the padded
        # input_ids are ignored according to the attention_mask
        pad_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        pad_masks = pad_sequence(masks, batch_first=True, padding_value=0)

        return {
            "input_ids": pad_input_ids,
            "attention_mask": pad_masks,
        }

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=pad_sequences,
        pin_memory=True,
    )


def format_calibration_data(
    tokenized_dataset: Dataset,
    num_calibration_samples: Optional[int] = None,
    do_shuffle: bool = True,
    collate_fn: Callable = default_data_collator,
    accelerator: Optional[Any] = None,
) -> torch.utils.data.DataLoader:
    """
    Creates a dataloader out of the calibration dataset split, trimming it to
    the desired number of calibration samples

    :param tokenized_dataset: dataset to convert to dataloader
    :param num_calibration_samples: number of data samples to convert
    :param do_shuffle: whether to shuffle the dataset before selecting calibration
    samples, true by default
    :param collate_fn: optional custom collate function, or use default
    :param accelerator: optional accelerator for if preparing in FSDP mode
    :return: list of trimmed calibration data tensors
    """
    safe_calibration_samples = len(tokenized_dataset)
    if num_calibration_samples is not None:
        safe_calibration_samples = min(len(tokenized_dataset), num_calibration_samples)
        if safe_calibration_samples != num_calibration_samples:
            LOGGER.warn(
                f"Requested {num_calibration_samples} calibration samples but "
                f"the provided dataset only has {safe_calibration_samples}. "
            )

    if do_shuffle:
        tokenized_dataset = tokenized_dataset.shuffle()
    tokenized_calibration = tokenized_dataset.select(range(safe_calibration_samples))

    dataloader_params = {
        "batch_size": 1,
        "sampler": RandomSampler(tokenized_calibration)
        if do_shuffle
        else SequentialSampler(tokenized_calibration),
        "collate_fn": collate_fn,
        "pin_memory": True,
    }

    calib_dataloader = DataLoader(tokenized_calibration, **dataloader_params)
    if accelerator:
        calib_dataloader = accelerator.prepare(calib_dataloader)

    return calib_dataloader


def get_raw_dataset(
    data_args,
    cache_dir: Optional[str] = None,
    streaming: Optional[bool] = False,
    **kwargs,
) -> Dataset:
    """
    Load the raw dataset from Hugging Face, using cached copy if available

    :param cache_dir: disk location to search for cached dataset
    :param streaming: True to stream data from Hugging Face, otherwise download
    :return: the requested dataset

    """
    raw_datasets = load_dataset(
        data_args.dataset,
        data_args.dataset_config_name,
        cache_dir=cache_dir,
        streaming=streaming,
        trust_remote_code=data_args.trust_remote_code_data,
        **kwargs,
    )
    return raw_datasets


def make_dataset_splits(
    datasets: Dict[str, DatasetType],
    do_train: bool = False,
    do_eval: bool = False,
    do_predict: bool = False,
    do_oneshot: bool = False,
) -> Dict[str, DatasetType]:
    """
    Restructures the datasets dictionary based on what tasks will be run
    (train, eval, predict)

    :param datasets: dictionary of processed datasets
    :param do_train: Whether to store the train dataset
    :param do_eval: Whether to store the validation dataset
    :param do_predict: Whether to store the test dataset
    :param do_oneshot: Whether to store the calibration dataset
    :return: Datasets to be used by the requested tasks
    """

    # handles case where all splits are contained in a single dataset
    if "all" in datasets and len(datasets) == 1:
        datasets = datasets.get("all")
        if isinstance(datasets, Dataset):
            datasets = {"train": datasets}

    train_split = eval_split = predict_split = calib_split = None

    if do_train:
        train_split = _get_split_with_fallbacks(
            datasets, "train", ["train"], strict=True
        )
    if do_eval:
        eval_split = _get_split_with_fallbacks(
            datasets, "evaluation", ["validation", "test"], strict=True
        )
    if do_predict:
        predict_split = _get_split_with_fallbacks(
            datasets, "prediction", ["test", "validation"], strict=True
        )
    if do_oneshot:
        calib_split = _get_split_with_fallbacks(
            datasets,
            "oneshot",
            ["calibration", "train", "test", "validation"],
            strict=False,
        )

        # remove labels from calibration dataset
        column_names = calib_split.column_names
        if isinstance(column_names, dict):
            column_names = sum(column_names.values(), [])
        if "labels" in column_names:
            calib_split = calib_split.remove_columns("labels")

    split_datasets = {
        "train": train_split,
        "validation": eval_split,
        "test": predict_split,
        "calibration": calib_split,
    }
    return split_datasets


def get_custom_datasets_from_path(path: str, ext: str = "json") -> Dict[str, str]:
    """
    Get a dictionary of custom datasets from a directory path. Support HF's load_dataset
     for local folder datasets https://huggingface.co/docs/datasets/loading

    This function scans the specified directory path for files with a
     specific extension (default is '.json').
    It constructs a dictionary where the keys are either subdirectory names or
     direct dataset names (depending on the directory structure)
    and the values are either file paths (if only one file exists with that name) or
     lists of file paths (if multiple files exist).

    :param path: The path to the directory containing the dataset files.
    :param ext: The file extension to filter files by. Default is 'json'.

    :return: A dictionary mapping dataset names to their file paths or lists of
     file paths.

    Example:
        dataset = get_custom_datasets_from_path("/path/to/dataset/directory", "json")

    Note:
        If datasets are organized in subdirectories, the function constructs the
         dictionary with lists of file paths.
        If datasets are found directly in the main directory, they are included with
         their respective names.

    Accepts:
        - path\
            train.json
            test.json
            val.json

        - path\
            train\
                data1.json
                data2.json
                ...
            test\
                ...
            val\
                ...

    """
    data_files = {}

    if any(filename.endswith(ext) for filename in os.listdir(path)):
        # If there are files with the given extension in the path
        for filename in os.listdir(path):
            if filename.endswith(ext):
                name, _ = os.path.splitext(filename)
                data_files[name] = os.path.join(path, filename)
    else:
        # If datasets are organized in subdirectories
        for root, dirs, files in os.walk(path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                dir_dataset = []
                for filename in os.listdir(dir_path):
                    if filename.endswith(ext):
                        file_path = os.path.join(dir_path, filename)
                        dir_dataset.append(file_path)
                if dir_dataset:
                    data_files[dir_name] = dir_dataset

    return transform_dataset_keys(data_files)


def transform_dataset_keys(data_files: Dict[str, Any]):
    """
    Transform dict keys to `train`, `val` or `test` for the given input dict
    if matches exist with the existing keys. Note that there can only be one
    matching file name.
    Ex. Folder(train_eval.json)          -> Folder(train.json)
        Folder(train1.json, train2.json) -> Same

    :param data_files: The dict where keys will be transformed
    """
    keys = set(data_files.keys())

    def transform_dataset_key(candidate: str) -> None:
        for key in keys:
            if candidate in key:
                if key == candidate:
                    return
                val = data_files.pop(key)
                data_files[candidate] = val

    def do_transform(candidate: str) -> bool:
        return sum(candidate in key for key in keys) == 1

    dataset_keys = ("train", "val", "test")
    for dataset_key in dataset_keys:
        if do_transform(dataset_key):
            transform_dataset_key(dataset_key)

    return data_files


def _get_split_with_fallbacks(
    datasets: Dict[str, DatasetType],
    task: str,
    fallbacks: List[str],
    strict: bool = True,
) -> DatasetType:
    assert len(fallbacks) > 0
    if len(datasets) <= 0:
        raise ValueError("Cannot get retrieve data from dataset with no splits")

    # check first choice
    first_choice = fallbacks[0]
    if first_choice in datasets:
        return datasets[first_choice]

    # last fallback is first available split
    if not strict:
        fallbacks.append(next(iter(datasets.keys())))

    # check fallbacks
    for fallback in fallbacks[1:]:
        if fallback in datasets:
            warnings.warn(
                f"{task} expects a {first_choice} dataset split, "
                f"falling back to {fallback}"
            )
            return datasets[fallback]

    raise ValueError(f"{task} expects at least one of {fallbacks} dataset splits")
