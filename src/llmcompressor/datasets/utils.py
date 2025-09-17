"""
Dataset utility functions for LLM compression workflows.

Provides helper functions for loading, processing, and formatting datasets used
in model compression pipelines. Handles dataset splitting, tokenization,
calibration data preparation, and dataloader creation for both training and
one-shot calibration workflows.
"""

import multiprocessing
import re
from typing import Any, Callable, Dict, List, Optional

import torch
from datasets import Dataset
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.data import default_data_collator

from llmcompressor.args import DatasetArguments
from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.typing import Processor


def get_processed_dataset(
    dataset_args: DatasetArguments,
    processor: Optional[Processor] = None,
    do_oneshot: bool = False,
    do_train: bool = True,
) -> Optional[Dict[str, Dataset]]:
    """
    Loads datasets for each flow based on dataset_args, stores a Dataset for each
    enabled flow in datasets
    :param dataset_args: DatasetArguments that contain dataset loading and
        processing params
    :param processor: processor or tokenizer to use for dataset tokenization
    :param do_oneshot: True for oneshot pathway
    :param do_train: True for train pathway
    :return: A dataset corresponding to either train or calibration (oneshot)
    """
    if dataset_args.dataset is None:
        logger.warning(
            "Running oneshot without calibration data. This is expected for "
            "weight-only and dynamic quantization"
        )
        return

    splits = dataset_args.splits
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
    registry_id = (
        dataset_args.dataset if isinstance(dataset_args.dataset, str) else "custom"
    )
    for split_name, split_str in splits.items():
        dataset = dataset_args.dataset
        if hasattr(dataset, "column_names") and "input_ids" in dataset.column_names:
            # dataset is already tokenized
            tokenized_datasets[split_name] = dataset
        else:
            # dataset needs to be tokenized
            dataset_manager = TextGenerationDataset.load_from_registry(
                registry_id,
                dataset_args=dataset_args,
                split=split_str,
                processor=processor,
            )
            tokenized_datasets[split_name] = dataset_manager(add_labels=do_train)

    return make_dataset_splits(
        tokenized_datasets,
        do_oneshot=do_oneshot,
        do_train=do_train,
    )


def get_calibration_dataloader(
    dataset_args: DatasetArguments,
    processor: Processor,
) -> torch.utils.data.DataLoader:
    """
    Get the dataloader used for oneshot calibration.
    :param dataset_args: DatasetArguments that contains the dataset parameters.
    :param processor: Processor or the tokenizer of the model.
    :return: PyTorch dataloader object that contains the calibration dataset.
    """
    if dataset_args.dataset is None:
        # weight-only quantization or dynamic quantization
        return

    datasets = get_processed_dataset(
        dataset_args=dataset_args,
        processor=processor,
        do_oneshot=True,
        do_train=False,
    )

    calibration_dataset = datasets.get("calibration")

    return format_calibration_data(
        tokenized_dataset=calibration_dataset,
        num_calibration_samples=dataset_args.num_calibration_samples,
        do_shuffle=dataset_args.shuffle_calibration_samples,
        collate_fn=dataset_args.data_collator,
    )


def format_calibration_data(
    tokenized_dataset: Dataset,
    num_calibration_samples: Optional[int] = None,
    do_shuffle: bool = True,
    collate_fn: Callable = default_data_collator,
) -> List[torch.Tensor]:
    """
    Creates a dataloader out of the calibration dataset split, trimming it to
    the desired number of calibration samples
    :param tokenized_dataset: dataset to convert to dataloader
    :param num_calibration_samples: number of data samples to convert
    :param do_shuffle: whether to shuffle the dataset before selecting calibration
        samples, true by default
    :param collate_fn: optional custom collate function, or use default
    :return: list of trimmed calibration data tensors
    """
    safe_calibration_samples = len(tokenized_dataset)
    if num_calibration_samples is not None:
        safe_calibration_samples = min(len(tokenized_dataset), num_calibration_samples)
        if safe_calibration_samples != num_calibration_samples:
            logger.warning(
                f"Requested {num_calibration_samples} calibration samples but "
                f"the provided dataset only has {safe_calibration_samples}. "
            )

    if do_shuffle:
        tokenized_dataset = tokenized_dataset.shuffle()
    tokenized_calibration = tokenized_dataset.select(range(safe_calibration_samples))

    MAX_DATALOADER_WORKERS = 8
    try:
        num_workers = min(MAX_DATALOADER_WORKERS, multiprocessing.cpu_count() // 2)
    except NotImplementedError:
        logger.warning(
            "Could not determine number of CPUs, defaulting to 0 dataloader workers."
        )
        num_workers = 0
    dataloader_params = {
        "batch_size": 1,
        "sampler": RandomSampler(tokenized_calibration)
        if do_shuffle
        else SequentialSampler(tokenized_calibration),
        "collate_fn": collate_fn,
        "pin_memory": True,
        "num_workers": num_workers,
    }

    calibration_dataloader = DataLoader(tokenized_calibration, **dataloader_params)

    return calibration_dataloader


def make_dataset_splits(
    tokenized_datasets: Dict[str, Any],
    do_oneshot: bool = True,
    do_train: bool = False,
) -> Dict[str, Dataset]:
    """
    Restructures the datasets dictionary based on what tasks will be run
    train
    :param tokenized_datasets: dictionary of processed datasets
    :param do_oneshot: Whether to store the calibration dataset
    :return: A dataset corresponding to either train or calibration (oneshot)
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
