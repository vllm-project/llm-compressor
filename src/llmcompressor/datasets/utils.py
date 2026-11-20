"""
Dataset utility functions for LLM compression workflows.

Provides helper functions for loading, processing, and formatting datasets used
in model compression pipelines. Handles dataset splitting, tokenization,
calibration data preparation, and dataloader creation for both training and
one-shot calibration workflows.
"""

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from typing import Any, Callable

import torch
from compressed_tensors.distributed import is_distributed
from compressed_tensors.modeling import attention as ct_attention
from datasets import Dataset
from loguru import logger
from torch import distributed as dist
from torch.utils.data import DataLoader, RandomSampler, Sampler
from transformers.data import (
    DataCollatorWithFlattening as HFDataCollatorWithFlattening,
)
from transformers.data import DataCollatorWithPadding, default_data_collator

from llmcompressor.args import DatasetArguments
from llmcompressor.transformers.data import TextGenerationDataset
from llmcompressor.typing import Processor

BS_WARNING_THRESHOLD = 16


def get_processed_dataset(
    dataset_args: DatasetArguments,
    processor: Processor | None = None,
) -> Dataset | None:
    """
    Loads dataset based on dataset_args.
    :param dataset_args: DatasetArguments that contain dataset loading and
        processing params
    :param processor: processor or tokenizer to use for dataset tokenization
    :return: A Dataset corresponding to the single split for calibration
    """
    if dataset_args.dataset is None:
        logger.warning(
            "Running oneshot without calibration data. This is expected for "
            "weight-only and dynamic quantization"
        )
        return

    splits = dataset_args.splits

    match splits:
        case None:
            split_str = None
        case str():
            split_str = splits
        case dict():
            if "calibration" in splits:
                split_str = splits["calibration"]
                if len(splits) > 1:
                    ignored_keys = set(splits.keys()) - {"calibration"}
                    logger.warning(
                        f"Ignoring extra keys in splits: {list(ignored_keys)}. "
                        "Only the 'calibration' split is used."
                    )
            else:
                raise ValueError(
                    "Passing `splits` as a dict is only supported when it contains a "
                    "`'calibration'` key during the deprecation period. "
                    "Please pass a split string instead."
                )

            logger.warning(
                "Passing `splits` as a dictionary is deprecated. "
                f"Extracted split string: '{split_str}'. "
                "Please pass `splits` as a string instead."
            )
        case list():
            split_str = splits[0] if len(splits) > 0 else None
            logger.warning(
                "Passing `splits` as a list is deprecated. "
                f"Using first element: '{split_str}'. "
                "Please pass `splits` as a string instead."
            )
        case _:
            raise ValueError(
                f"Invalid splits type: {type(splits)}. Expected a split string "
                "or the deprecated `{'calibration': ...}` form."
            )

    # default to custom dataset if dataset provided isn't a string
    registry_id = (
        dataset_args.dataset if isinstance(dataset_args.dataset, str) else "custom"
    )

    dataset = dataset_args.dataset
    if hasattr(dataset, "column_names") and "input_ids" in dataset.column_names:
        # dataset is already tokenized
        return dataset
    else:
        # dataset needs to be tokenized
        dataset_manager = TextGenerationDataset.load_from_registry(
            registry_id,
            dataset_args=dataset_args,
            split=split_str,
            processor=processor,
        )
        dataset = dataset_manager()

        # If no split was specified, a DatasetDict format is typically returned.
        # Fallback to the 'train' split for backward compatibility.
        if not isinstance(dataset, Dataset):
            if "train" in dataset:
                logger.warning(
                    "No split was specified, but a multi-split dataset was loaded. "
                    "Falling back to the 'train' split for calibration."
                )
                dataset = dataset["train"]
            else:
                raise ValueError(
                    "No split specified and 'train' split not found in dataset. "
                    "Please specify `splits` explicitly."
                )

        return dataset


def get_calibration_dataloader(
    dataset_args: DatasetArguments,
    processor: Processor,
) -> DataLoader | None:
    """
    Get the dataloader used for oneshot calibration.

    If dataset_args.dataset is already a PyTorch DataLoader,
    it is returned directly, bypassing dataset loading and tokenization.

    :param dataset_args: DatasetArguments that contains the dataset parameters.
    :param processor: Processor or the tokenizer of the model.
    :return: PyTorch dataloader object that contains the calibration
        dataset, or None for data-free flows.
    """
    if dataset_args.dataset is None:
        # weight-only quantization or dynamic quantization
        return None

    if isinstance(dataset_args.dataset, DataLoader):
        return dataset_args.dataset

    calibration_dataset = get_processed_dataset(
        dataset_args=dataset_args,
        processor=processor,
    )

    if calibration_dataset is None:
        return None

    return format_calibration_data(dataset_args, calibration_dataset, processor)


def format_calibration_data(
    args: DatasetArguments,
    tokenized_dataset: Dataset,
    processor: Processor,
) -> DataLoader:
    # Pin memory only when using workers (saves RAM for low-memory users when
    # num_workers=0; when num_workers>0, pin_memory speeds CPU->GPU transfer)
    num_workers = args.dataloader_num_workers
    pin_memory = torch.accelerator.is_available() and num_workers > 0
    # persistent_workers avoids worker respawn between epochs (only when
    # num_workers > 0). prefetch_factor is left at DataLoader default (2).
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(
        tokenized_dataset,
        batch_size=get_initial_batch_size(args),
        sampler=_make_sampler(args, tokenized_dataset),
        collate_fn=_make_collate_fn(args, processor),
        pin_memory=pin_memory,
        num_workers=num_workers,
        **kwargs,
    )


def _make_collate_fn(args: DatasetArguments, processor: Processor) -> Callable:
    if isinstance(args.data_collator, Callable):
        return args.data_collator

    if args.data_collator == "truncation":
        if get_initial_batch_size(args) > BS_WARNING_THRESHOLD:
            logger.warning(
                f"Calibrating with batch sizes greater than {BS_WARNING_THRESHOLD} and "
                "`data_collator='truncation'` can lead to significant portions of the "
                "calibration dataset being deleted via truncation. Please consider "
                "reducing the calibration batch size or using filtering the dataset "
                "to use more uniform sequence lengths"
            )

        return DataCollatorWithTruncation(args.max_seq_length)

    elif args.data_collator == "padding":
        if get_initial_batch_size(args) > BS_WARNING_THRESHOLD:
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

        if args.max_seq_length is not None:
            logger.warning(
                "Cannot use `data_collator='padding'` with `max_seq_length`. Ignoring "
                "truncation specified by `max_seq_length`"
            )

        return DataCollatorWithPadding(tokenizer)

    elif args.data_collator == "flatten":
        return DataCollatorWithFlattening(args.max_seq_length)

    else:
        raise ValueError(f"Unknown data collator {args.data_collator}")


def _is_dist_and_same_ds(dataset: Dataset) -> bool:
    if not is_distributed():
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
    batch_size = get_initial_batch_size(args)

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
        # packing neither pads nor truncates, so the order of samples costs nothing
        if batch_size > 1 and args.data_collator != "flatten":
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


class DataCollatorWithTruncation:
    def __init__(self, max_seq_length: int | None = None):
        self.max_seq_length = max_seq_length

    def __call__(
        self, features: list[dict[str, Any]], return_tensors: str = "pt"
    ) -> dict[str, Any]:
        for key in (
            "input_ids",
            "labels",
            "attention_mask",
            "loss_mask",
            "mm_token_type_ids",
        ):
            if any(key not in feature for feature in features):
                continue

            min_len = min(len(feature[key]) for feature in features)
            if self.max_seq_length is not None:
                min_len = min(min_len, self.max_seq_length)
            for feature in features:
                feature[key] = feature[key][:min_len]

        return default_data_collator(features, return_tensors)


class DataCollatorWithFlattening:
    """
    Concatenate the samples of a batch into one row, without padding or truncating
    them to a common length. `position_ids` restart at 0 for each sample, which
    transformers uses to keep attention within each sample when there is no
    `attention_mask` and no KV cache, as during calibration. Unlike
    `transformers.DataCollatorWithFlattening`, `loss_mask` is kept, so that
    `use_loss_mask` still masks tokens.

    Tokens with an `attention_mask` of 0 are dropped, and `token_type_ids` which are
    all 0 are dropped. Only text samples are supported: other features raise a
    ValueError, as does a `loss_mask` that only some samples of a batch have.

    Packing is only exact for models which build attention masks from
    `position_ids`, and is only memory efficient with attention implementations
    which don't build the full attention mask (flex or flash attention). See
    `check_flattening_support`.

    :param max_seq_length: truncate each sample to this many tokens, if set
    """

    SUPPORTED_KEYS = {"input_ids", "attention_mask", "labels", "loss_mask"}

    def __init__(self, max_seq_length: int | None = None):
        self.max_seq_length = max_seq_length

    def __call__(
        self, features: list[dict[str, Any]], return_tensors: str = "pt"
    ) -> dict[str, torch.Tensor]:
        features = [_drop_zero_token_type_ids(feature) for feature in features]
        unsupported = {key for f in features for key in f} - self.SUPPORTED_KEYS
        if unsupported:
            raise ValueError(
                'data_collator="flatten" only supports text samples, but samples '
                f"have {', '.join(sorted(unsupported))}"
            )

        has_loss_mask = ["loss_mask" in feature for feature in features]
        if any(has_loss_mask) and not all(has_loss_mask):
            raise ValueError(
                "Only some calibration samples have a `loss_mask`, so tokens of the "
                "others could not be masked"
            )
        keep_loss_mask = all(has_loss_mask)
        input_ids, position_ids, loss_mask = [], [], []
        for feature in features:
            ids = list(feature["input_ids"])
            mask = list(feature["loss_mask"]) if keep_loss_mask else None
            if "attention_mask" in feature:
                kept = [i for i, m in enumerate(feature["attention_mask"]) if m]
                ids = [ids[i] for i in kept]
                mask = [mask[i] for i in kept] if mask is not None else None
            if self.max_seq_length is not None:
                ids = ids[: self.max_seq_length]

            input_ids.extend(ids)
            position_ids.extend(range(len(ids)))
            if mask is not None:
                loss_mask.extend(mask[: len(ids)])

        batch = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "position_ids": torch.tensor([position_ids], dtype=torch.long),
        }
        if keep_loss_mask:
            batch["loss_mask"] = torch.tensor([loss_mask])
        return batch


# collators which pack a batch into one row, see `TokenBudgetBatchSampler`
PACKING_COLLATORS = (DataCollatorWithFlattening, HFDataCollatorWithFlattening)

# model types which build their attention masks from `position_ids`, so that packed
# samples don't attend to each other. Each was checked by comparing two samples run
# separately with the same samples packed into one row, with sdpa and eager attention
# where the model supports them.
# Others, such as Falcon, Bloom, MPT and OPT (masks without `position_ids`) or
# Qwen3-Next (linear attention), mix the samples of a row
PACKING_MODEL_TYPES = frozenset(
    {
        "apertus",
        "cohere",
        "cohere2",
        "ernie4_5",
        "exaone4",
        "gemma",
        "gemma2",
        "gemma3_text",
        "glm",
        "glm4",
        "gpt2",
        "gpt_neox",
        "gptj",
        "granite",
        "granitemoe",
        "hunyuan_v1_dense",
        "llama",
        "mistral",
        "mixtral",
        "olmo2",
        "olmoe",
        "phi",
        "phi3",
        "phimoe",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "seed_oss",
        "smollm3",
        "stablelm",
        "starcoder2",
    }
)


def check_flattening_support(model: torch.nn.Module) -> None:
    """
    Check that calibrating `model` on batches packed by `DataCollatorWithFlattening`
    gives the same result as calibrating on each sample, which holds for the model
    types in `PACKING_MODEL_TYPES`. Eager and sdpa attention are exact but build a
    seq_len x seq_len mask for each packed row, so a warning suggests flex or flash
    attention

    :param model: model being calibrated
    :raises ValueError: if packing is not known to keep samples apart in this model
    """
    config = getattr(model, "config", None)
    text_config = (
        config.get_text_config() if hasattr(config, "get_text_config") else config
    )
    model_type = getattr(text_config, "model_type", None)
    if model_type not in PACKING_MODEL_TYPES:
        raise ValueError(
            'data_collator="flatten" packs samples into one row, which only keeps '
            "samples apart in models which build their attention masks from "
            f"`position_ids`. Model type {model_type!r} is not known to, so its "
            "samples could attend to, or carry state into, each other. Use the "
            "truncation or padding data collator instead"
        )

    implementation = get_attention_implementation(model)
    if not supports_packed_attention(implementation):
        logger.warning(
            f'data_collator="flatten" with {implementation} attention builds the '
            "full attention mask of every packed row, which needs memory quadratic in "
            "its length. Loading the model with attn_implementation='flex_attention' "
            "or 'flash_attention_2' avoids it"
        )


def get_attention_implementation(model: torch.nn.Module) -> str:
    """
    :param model: model being calibrated
    :return: the attention implementation the model runs, looking through the hooked
        implementation compressed-tensors installs for attention and KV cache
        quantization
    """
    implementation = (
        getattr(getattr(model, "config", None), "_attn_implementation", None) or "eager"
    )
    if implementation == getattr(ct_attention, "HOOKED_ATTENTION_NAME", None):
        implementation = ct_attention.QuantizedAttentionImpl._original_impl
    return implementation


def supports_packed_attention(implementation: str) -> bool:
    """
    :param implementation: attention implementation, see `get_attention_implementation`
    :return: whether it attends within packed samples without building the full
        attention mask of the row (flex or flash attention, including hub kernels)
    """
    return "flash" in implementation or "flex" in implementation


def _drop_zero_token_type_ids(feature: dict[str, Any]) -> dict[str, Any]:
    """Drop `token_type_ids` which only mark one segment, as text tokenizers emit"""
    token_type_ids = feature.get("token_type_ids")
    if token_type_ids is not None and not any(token_type_ids):
        return {key: value for key, value in feature.items() if key != "token_type_ids"}
    return feature


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
        num_samples: int | None = None,
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


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """
    Group samples into batches of at most `max_tokens` tokens, where a batch counts as
    its number of samples times its longest sample (its padded size, which also bounds
    a truncated batch).

    The samples are those drawn once from `sampler`, so that the same subset of the
    dataset is calibrated on as with batch size 1. They are grouped by descending
    length so that samples of similar length share a batch. A sample only joins a
    batch if it is at least `1 - max_length_spread` times as long as the batch's
    longest sample, which bounds the tokens that padding adds or truncation removes. A
    sample longer than `max_tokens` is given a batch of its own.

    With `packed=True`, for collators which concatenate a batch's samples into one
    row (see `DataCollatorWithFlattening`), a batch counts as the sum of its samples'
    lengths. Nothing is padded or truncated, so samples keep the sampler's order and
    fill each batch in turn.

    Batches are then ordered by their earliest sample in the sampler's order, and the
    samples of a batch keep that order, so that statistics which depend on the order
    of samples (such as moving averages) don't see them sorted by length.

    :param sampler: sampler, or sequence of dataset indices, which selects and orders
        the calibration samples
    :param lengths: length of each sampled sample, by dataset index
    :param max_tokens: maximum number of tokens per batch, counting padding
    :param max_length_spread: largest fraction by which a sample may be shorter than
        the longest sample of its batch. Not used when `packed`
    :param packed: whether batches are packed into one row rather than padded or
        truncated
    """

    def __init__(
        self,
        sampler: Iterable[int],
        lengths: Sequence[int] | Mapping[int, int],
        max_tokens: int,
        max_length_spread: float = 0.1,
        packed: bool = False,
    ):
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        sampled = list(sampler)
        position = {index: order for order, index in enumerate(sampled)}

        if packed:
            indices = sampled
        else:
            # sorting is stable, so samples of equal length keep the sampler's order
            indices = sorted(sampled, key=lambda index: lengths[index], reverse=True)

        self.batches: list[list[int]] = []
        batch: list[int] = []
        longest, total = 0, 0
        for index in indices:
            length = lengths[index]
            if packed:
                is_full = total + length > max_tokens
            else:
                is_full = (len(batch) + 1) * longest > max_tokens or (
                    length < (1 - max_length_spread) * longest
                )
            if batch and is_full:
                self.batches.append(batch)
                batch, total = [], 0

            if not batch:
                longest = length
            batch.append(index)
            total += length

        if batch:
            self.batches.append(batch)

        for batch in self.batches:
            batch.sort(key=position.__getitem__)
        self.batches.sort(key=lambda batch: position[batch[0]])

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def get_initial_batch_size(args: DatasetArguments) -> int:
    """
    :param args: dataset arguments
    :return: batch size of the calibration dataloader as it is built. With
        `batch_size="auto"`, batches are sized later by the sequential pipeline
        (see `batch_by_tokens`), so the dataloader starts with batch size 1
    """
    return 1 if args.batch_size == "auto" else args.batch_size


def batch_by_tokens(dataloader: DataLoader, max_tokens: int) -> DataLoader:
    """
    Rebuild a calibration dataloader so that its batches hold up to `max_tokens`
    tokens each, using `TokenBudgetBatchSampler`. The dataset, collate function and
    loading options are kept, so batches have the same keys as before

    :param dataloader: calibration dataloader with batch size 1
    :param max_tokens: maximum number of tokens per batch, counting padding
    :return: the rebuilt dataloader, or `dataloader` itself if it is not a
        `DataLoader` or the lengths of its samples cannot be read
    """
    indices = _draw_sample_indices(dataloader)
    lengths = _get_sample_lengths(dataloader, indices) if indices is not None else None
    if lengths is None:
        logger.warning(
            'batch_size="auto" needs a DataLoader over a map-style dataset whose '
            "samples have `input_ids`. Calibrating with batch_size=1"
        )
        return dataloader

    packed = isinstance(dataloader.collate_fn, PACKING_COLLATORS)
    batch_sampler = TokenBudgetBatchSampler(indices, lengths, max_tokens, packed=packed)
    batch_sizes = [len(batch) for batch in batch_sampler.batches]
    logger.info(
        f'batch_size="auto": {sum(batch_sizes)} samples in {len(batch_sizes)} '
        f"batch(es) of {min(batch_sizes)} to {max(batch_sizes)} samples"
    )

    kwargs: dict[str, Any] = {}
    if dataloader.num_workers > 0:
        kwargs["persistent_workers"] = dataloader.persistent_workers
        kwargs["prefetch_factor"] = dataloader.prefetch_factor
    return DataLoader(
        dataloader.dataset,
        batch_sampler=batch_sampler,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        num_workers=dataloader.num_workers,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        **kwargs,
    )


def _draw_sample_indices(dataloader: Any) -> list[int] | None:
    """
    Draw the dataset indices which a batch size 1 dataloader would calibrate on,
    without advancing the global random state or the dataloader's own generators,
    which calibration continues to draw from

    :param dataloader: calibration dataloader
    :return: dataset indices in the sampler's order, or None if `dataloader` is not
        a `DataLoader` over a map-style dataset
    """
    if not isinstance(dataloader, DataLoader) or not isinstance(
        dataloader.dataset, Sized
    ):
        return None

    sampler = dataloader.sampler
    candidates = (dataloader.generator, getattr(sampler, "generator", None))
    generators = {id(g): g for g in candidates if isinstance(g, torch.Generator)}
    states = {key: generator.get_state() for key, generator in generators.items()}
    try:
        with torch.random.fork_rng(devices=[]):
            return list(sampler)
    finally:
        for key, generator in generators.items():
            generator.set_state(states[key])


def _get_sample_lengths(
    dataloader: DataLoader, indices: list[int]
) -> dict[int, int] | None:
    """
    Read the number of tokens of the sampled samples, after the truncation which the
    dataloader's collate function applies

    :param dataloader: calibration dataloader over a map-style dataset
    :param indices: dataset indices of the sampled samples
    :return: length of each sampled sample by dataset index, or None if they cannot
        be read
    """
    dataset = dataloader.dataset
    try:
        if isinstance(dataset, Dataset) and "input_ids" in dataset.column_names:
            unique = sorted(set(indices))
            column = dataset.select(unique)["input_ids"]
            lengths = {index: len(ids) for index, ids in zip(unique, column)}
        else:
            lengths = {index: len(dataset[index]["input_ids"]) for index in indices}
    except (KeyError, TypeError, IndexError):
        return None

    collate_fn = dataloader.collate_fn
    max_seq_length = getattr(collate_fn, "max_seq_length", None)
    truncating = (DataCollatorWithTruncation, DataCollatorWithFlattening)
    if isinstance(collate_fn, truncating) and max_seq_length:
        lengths = {index: min(n, max_seq_length) for index, n in lengths.items()}

    return lengths


def get_rank_partition(split: str, num_samples: int) -> str:
    """
    Utility for splitting data in a distributed setting and
    also works in non-distributed setting

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

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    start, end = _get_partition_start_end(num_samples, rank, world_size)
    return f"{split}[{start}:{end}]"
