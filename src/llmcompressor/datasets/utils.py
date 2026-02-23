"""
Dataset utility functions for LLM compression workflows.

Provides helper functions for loading, processing, and formatting datasets used
in model compression pipelines. Handles dataset splitting, tokenization,
calibration data preparation, and dataloader creation for both training and
one-shot calibration workflows.
"""

import math
import re
from collections.abc import Iterator, Sized
from typing import Any, Callable, Optional

import torch
from datasets import Dataset
from loguru import logger
from torch import distributed as dist
from torch.utils.data import DataLoader, RandomSampler, Sampler
from transformers.data import DataCollatorWithPadding, default_data_collator

from llmcompressor.args import DatasetArguments
from llmcompressor.transformers.data import TextGenerationDataset
from llmcompressor.typing import Processor

BS_WARNING_THRESHOLD = 16


def get_processed_dataset(
    dataset_args: DatasetArguments,
    processor: Processor | None = None,
    do_oneshot: bool = False,
    do_train: bool = True,
) -> dict[str, Dataset] | None:
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
        split_name_match = re.match(r"(\w*)\[.*\]", inp_str)
        if split_name_match is not None:
            return split_name_match.group(1)
        return inp_str

    match splits:
        case None:
            splits = {"all": None}
        case str():
            splits = {_get_split_name(splits): splits}
        case list():
            splits = {_get_split_name(s): s for s in splits}
        case dict():
            pass
        case _:
            raise ValueError(f"Invalid splits type: {type(splits)}")

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

    return format_calibration_data(dataset_args, calibration_dataset, processor)


def format_calibration_data(
    args: DatasetArguments,
    tokenized_dataset: Dataset,
    processor: Processor,
) -> DataLoader:
    # Pin memory only when using workers (saves RAM for low-memory users when
    # num_workers=0; when num_workers>0, pin_memory speeds CPU->GPU transfer)
    num_workers = args.dataloader_num_workers
    pin_memory = torch.cuda.is_available() and num_workers > 0
    # persistent_workers avoids worker respawn between epochs (only when
    # num_workers > 0). prefetch_factor is left at DataLoader default (2).
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        sampler=_make_sampler(args, tokenized_dataset),
        collate_fn=_make_collate_fn(args, processor),
        pin_memory=pin_memory,
        num_workers=num_workers,
        **kwargs,
    )


def make_dataset_splits(
    tokenized_datasets: dict[str, Any],
    do_oneshot: bool = True,
    do_train: bool = False,
) -> dict[str, Dataset]:
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


def _make_collate_fn(args: DatasetArguments, processor: Processor) -> Callable:
    if isinstance(args.data_collator, Callable):
        return args.data_collator

    if args.data_collator == "truncation":
        if args.batch_size > BS_WARNING_THRESHOLD:
            logger.warning(
                f"Calibrating with batch sizes greater than {BS_WARNING_THRESHOLD} and "
                "`data_collator='truncation'` can lead to significant portions of the "
                "calibration dataset being deleted via truncation. Please consider "
                "reducing the calibration batch size or using filtering the dataset "
                "to use more uniform sequence lengths"
            )

        return data_collator_with_truncation

    elif args.data_collator == "padding":
        if args.batch_size > BS_WARNING_THRESHOLD:
            logger.warning(
                f"Calibrating with batch sizes greater than {BS_WARNING_THRESHOLD} and "
                "`data_collator='padding'` can lead to excess token used for padding, "
                "which slows down calibration time and calibrates on padding tokens not"
                " seen at runtime. Please consider reducing the calibration batch size "
                "or using filtering the dataset to use more uniform sequence lengths"
            )

        tokenizer = getattr(processor, "tokenizer", processor)
        if tokenizer.pad_token is None or tokenizer.pad_token_id < 0:
            logger.debug("Could not find padding token. Setting PAD token to EOS token")
            tokenizer.pad_token = tokenizer.eos_token

        return DataCollatorWithPadding(tokenizer)

    else:
        raise ValueError(f"Unknown data collator {args.data_collator}")


def _is_dist_and_same_ds(dataset: Dataset) -> bool:
    if not dist.is_initialized():
        return False

    assert len(dataset) > 0, (
        "Dataset must have at least one sample on each"
        f"device but got None for rank={dist.get_rank()}"
    )

    # use _fingerprint if it exists, otherwise hash the first sample.
    # This isn't perfect but should work in most cases
    local_hash = getattr(dataset, "_fingerprint", str(abs(hash(str(dataset[0])))))

    all_hashes = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_hashes, local_hash)

    return all(local_hash == other_hash for other_hash in all_hashes)


def _get_partition_start_end(
    num_samples: int, index: int, num_partitions: int
) -> tuple[int, int]:
    # num_samples / num_partitions is average samples per partition
    # we multiply this number with the partition indices to get partition bounds
    # note that final partition has index+1 == num_partitions so it will
    # always get all samples
    start = math.floor(num_samples * (index / num_partitions))
    end = math.floor(num_samples * ((index + 1) / num_partitions))
    return start, end


def _make_sampler(args: DatasetArguments, dataset: Dataset) -> Sampler:
    num_samples = args.num_calibration_samples
    shuffle = args.shuffle_calibration_samples
    batch_size = args.batch_size

    # detect whether we're in a distributed setting
    # but all ranks have the same dataset.
    if _is_dist_and_same_ds(dataset):
        logger.info(
            "Detected distributed setting with identical datasets across ranks. "
            "partitioning dataset across ranks."
        )
        num_samples = num_samples or len(dataset)
        start, end = _get_partition_start_end(
            num_samples, dist.get_rank(), dist.get_world_size()
        )
        dataset = dataset.select(range(start, end))

    if num_samples is not None and num_samples > len(dataset):
        logger.warning(
            f"Requested {num_samples} samples but the provided dataset only has "
            f"{len(dataset)} samples."
        )
        num_samples = len(dataset)

    if shuffle:
        if batch_size > 1:
            logger.warning(
                "Shuffling the dataset can lead to unoptimal batching for sequence "
                "lengths non-uniform sizes. When collating with truncation, this will "
                "delete a large number of tokens. When collating with padding, this "
                "will add a large number of padding tokens.\n\nPlease consider calling "
                "`oneshot` with `batch_size=1`"
            )

        return RandomSampler(dataset, num_samples=num_samples)
    else:
        return LengthAwareSampler(
            dataset, num_samples=num_samples, batch_size=batch_size
        )


def data_collator_with_truncation(
    features: list[dict[str, Any]], return_tensors: str = "pt"
) -> dict[str, Any]:
    for key in ("input_ids", "labels", "attention_mask", "loss_mask"):
        if any(key not in feature for feature in features):
            continue

        min_len = min(len(feature[key]) for feature in features)
        for feature in features:
            feature[key] = feature[key][:min_len]

    return default_data_collator(features, return_tensors)


class LengthAwareSampler(Sampler[int]):
    """
    Sample data in order of descending sequence length. Relies on `input_ids` or
    `decoder_input_ids` column existing in dataset

    :param data_source: dataset containing a `input_ids` or `decoder_input_ids` column
    :param num_samples: Maximum number of samples to sample. Shorted sequence lengths
        are truncated first
    """

    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Dataset,
        num_samples: Optional[int] = None,
        batch_size: int = 1,
    ) -> None:
        self.data_source = data_source
        self._num_samples = num_samples or len(data_source)
        self.batch_size = batch_size

        if "input_ids" in data_source.column_names:
            feature_name = "input_ids"
        elif "decoder_input_ids" in data_source.column_names:
            feature_name = "decoder_input_ids"
        else:
            logger.warning(f"Could not find input ids in {data_source.column_names}")
            self.order = range(len(data_source))
            return

        lengths = [len(sample) for sample in data_source[feature_name]]
        self.order = torch.argsort(torch.tensor(lengths), descending=True).tolist()
        self._calculate_and_log_batch_stats(lengths)

    def _calculate_and_log_batch_stats(self, lengths: list[int]):
        if self.batch_size == 1:
            return

        logger.debug(
            "LengthAwareSampler: Calculating batch statistics for "
            f"{self.num_samples} samples with batch size {self.batch_size}"
        )

        sorted_lengths = [lengths[i] for i in self.order][: self.num_samples]
        total_tokens_removed = 0
        total_tokens_added = 0

        for i in range(0, self.num_samples, self.batch_size):
            batch_lengths = sorted_lengths[i : i + self.batch_size]
            if not batch_lengths:
                continue

            shortest_in_batch = min(batch_lengths)
            longest_in_batch = max(batch_lengths)
            tokens_removed = sum(lgth - shortest_in_batch for lgth in batch_lengths)
            tokens_added = sum(longest_in_batch - lgth for lgth in batch_lengths)

            total_tokens_removed += tokens_removed
            total_tokens_added += tokens_added

        if total_tokens_removed > 0 or total_tokens_added > 0:
            logger.debug(
                f"LengthAwareSampler: Total token overhead - "
                f"removed (truncation): {total_tokens_removed}, "
                f"added (padding): {total_tokens_added}"
            )

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        return iter(self.order[: self._num_samples])

    def __len__(self) -> int:
        return self._num_samples


def get_rank_partition(split: str, num_samples: int) -> str:
    """
    Utility for splitting data in a distributed setting

    :param split: the split string to partition, e.g. "train"
    :param num_samples: the total number of samples in the dataset to partition
    :return: a partitioned split string

    Usage example:

    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"
    NUM_CALIBRATION_SAMPLES = 256

    split = get_rank_partition(DATASET_SPLIT, NUM_CALIBRATION_SAMPLES)
    ds = load_dataset(
        DATASET_ID,
        split=split,
    )

    for S samples and D devices, when S is not perfectly divisible by D,
    we give each device at least S//D samples and distribute
    the remaining samples as evenly as possible across all devices
    """
    assert (
        "[" not in split
    ), "Split string should not already contain partitioning brackets"

    start, end = _get_partition_start_end(
        num_samples, dist.get_rank(), dist.get_world_size()
    )
    return f"{split}[{start}:{end}]"
