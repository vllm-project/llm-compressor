from typing import Callable, Dict, List, Optional

import torch
from datasets import Dataset
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.data import default_data_collator

from llmcompressor.args import DatasetArguments
from llmcompressor.transformers.finetune.data import TextGenerationDataset
from llmcompressor.typing import DatasetType, Processor


def get_processed_dataset(
    dataset_args: DatasetArguments, processor: Processor, add_labels: bool
) -> Optional[Dict[str, Dataset]]:
    """
    Loads datasets for each flow based on dataset_args, stores a Dataset for each
    enabled flow in datasets
    :param dataset_args: DatasetArguments that contain dataset loading and
        processing params
    :param processor: processor or tokenizer to use for dataset tokenization
    :param add_labels: set to False for calibration, True for train and evaluation
    :return: tokenized dataset
    """
    dataset = dataset_args.dataset
    assert isinstance(dataset, (str, DatasetType))

    registry_id = dataset if isinstance(dataset, str) else "custom"
    dataset_manager = TextGenerationDataset.load_from_registry(
        registry_id,
        dataset_args=dataset_args,
        split=dataset_args.split,
        processor=processor,
    )
    return dataset_manager(add_labels=add_labels)


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
    dataset = dataset_args.dataset

    # check if dataset is already processed/tokenized
    if hasattr(dataset, "column_names") and "input_ids" in dataset.column_names:
        return

    # process dataset
    calibration_dataset = get_processed_dataset(
        dataset_args, processor, add_labels=False
    )

    # create dataloader
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
            logger.warn(
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

    calibration_dataloader = DataLoader(tokenized_calibration, **dataloader_params)

    return calibration_dataloader
