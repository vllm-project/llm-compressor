import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.data import default_data_collator

LOGGER = logging.getLogger(__name__)
LABELS_MASK_VALUE = -100

__all__ = [
    "format_calibration_data",
    "get_raw_dataset",
    "make_dataset_splits",
    "get_custom_datasets_from_path",
    "get_calibration_dataloader",
]


def format_calibration_data(
    tokenized_dataset: Dataset,
    num_calibration_samples: Optional[int] = None,
    do_shuffle: bool = True,
    collate_fn: Callable = default_data_collator,
    accelerator: Optional[Any] = None,
) -> List[torch.Tensor]:
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
    tokenized_datasets: Dict[str, Any],
    do_train: bool = False,
    do_oneshot: bool = False,
) -> Dict[str, Dataset]:
    """
    Restructures the datasets dictionary based on what tasks will be run
    train

    :param tokenized_datasets: dictionary of processed datasets
    :param do_oneshot: Whether to store the calibration dataset

    :return: Datasets to be used by the requested tasks
    """

    # handles case where all splits are contained in a single dataset
    if "all" in tokenized_datasets and len(tokenized_datasets) == 1:
        tokenized_datasets = tokenized_datasets.get("all")
        if isinstance(tokenized_datasets, Dataset):
            tokenized_datasets = {"train": tokenized_datasets}

    train_split = calib_split = None

    if do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_split = tokenized_datasets["train"]
    if do_oneshot:
        calib_split = tokenized_datasets.get("calibration")
        if calib_split is None:
            if "train" not in tokenized_datasets:
                raise ValueError("--do_oneshot requires a calibration dataset")
            calib_split = tokenized_datasets["train"]

    split_datasets = {
        "train": train_split,
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
    Ex. Folder(train_foo.json)           -> Folder(train.json)
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


def get_calibration_dataloader(
    data_args,
    processor,
    add_labels: bool = False,  # for oneshot
    do_oneshot=True,
) -> torch.utils.data.DataLoader:
    """
    Loads datasets for each flow based on data_args, stores a Dataset for each
    enabled flow in self.datasets

    :param processor: processor or tokenizer to use for dataset tokenization
    :param add_labels: if True, add labels column to dataset splits
    """
    if data_args.dataset is None:
        logger.info(
            "Running oneshot without calibration data. This is expected for "
            "weight-only and dynamic quantization"
        )
        return

    splits = data_args.splits
    tokenized_datasets = {}

    def _get_split_name(inp_str):
        # strip out split name, for ex train[60%:] -> train
        match = re.match(r"(\w*)\[.*\]", inp_str)
        if match is not None:
            return match.group(1)
        return inp_str

    if splits is None:
        splits = {"all": None}
    elif isinstance(splits, str):
        splits = {_get_split_name(splits): splits}
    elif isinstance(splits, List):
        splits = {_get_split_name(s): s for s in splits}

    # default to custom dataset if dataset provided isn't a string
    registry_id = data_args.dataset if isinstance(data_args.dataset, str) else "custom"
    for split_name, split_str in splits.items():
        dataset = data_args.dataset
        if hasattr(dataset, "column_names") and "input_ids" in dataset.column_names:
            # dataset is already tokenized
            tokenized_datasets[split_name] = dataset
        else:
            # dataset needs to be tokenized
            from llmcompressor.transformers.finetune.data.base import (
                TextGenerationDataset,
            )

            dataset_manager = TextGenerationDataset.load_from_registry(
                registry_id,
                data_args=data_args,
                split=split_str,
                processor=processor,
            )
            tokenized_datasets[split_name] = dataset_manager(add_labels=add_labels)

    datasets = make_dataset_splits(
        tokenized_datasets,
        do_oneshot=do_oneshot,
    )

    calibration_dataset = datasets.get("calibration")

    return format_calibration_data(
        tokenized_dataset=calibration_dataset,
        num_calibration_samples=data_args.num_calibration_samples,
        do_shuffle=data_args.shuffle_calibration_samples,
        collate_fn=data_args.data_collator,
    )
