from functools import cached_property
from typing import Any, Callable, Union

from compressed_tensors.registry import RegistryMixin
from datasets import Dataset, DatasetDict, IterableDataset
from loguru import logger
from transformers import AutoProcessor

from llmcompressor.transformers.finetune.data.data_args import DataTrainingArguments
from llmcompressor.transformers.finetune.data.data_helpers import (
    LABELS_MASK_VALUE,
    get_custom_datasets_from_path,
    get_raw_dataset,
)
from llmcompressor.transformers.utils.preprocessing_functions import (
    PreprocessingFunctionRegistry,
)
from llmcompressor.utils import import_from_path

DatasetType = Union[Dataset, DatasetDict, IterableDataset]


class TextGenerationDataset(RegistryMixin):
    """
    Base class for text datasets, handles tokenization and dataset splits

    :param text_column: name of column corresponding to text in the dataset
    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param tokenizer: tokenizer to use on dataset
    """

    # used to mask out the prompt so prompt tokens do not contribute to training loss
    PROMPT_KEY = "prompt"

    # TODO: not sure how to handle the prompt stuff best. Specifically w.r.t.
    """
    dataset = self.processor(**dataset)

    if dataset includes the PROMPT_KEY
    """

    def __init__(
        self,
        data_args: DataTrainingArguments,
        split: str,
        processor: "AutoProcessor",
    ):
        self.data_args = data_args
        self.split = split
        self.processor = processor

        # get tokenizer
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

        # fill in pad token
        if self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # configure sequence length
        max_seq_length = data_args.max_seq_length
        if data_args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({max_seq_length}) is larger than "
                f"the maximum length for model ({self.tokenizer.model_max_length}). "
                f"Using max_seq_length={self.tokenizer.model_max_length}."
            )
        self.max_seq_length = min(
            data_args.max_seq_length, self.tokenizer.model_max_length
        )

    def __call__(self, add_labels: bool) -> DatasetType:
        dataset = self.data_args.dataset

        # load dataset
        if isinstance(dataset, str):
            dataset = self.load_dataset()

        # preprocess
        if self.preprocess is not None:
            dataset = self.map(
                dataset,
                self.preprocess,
                batched=False,
                remove_columns=dataset.column_names,
                num_proc=self.data_args.preprocessing_num_workers,
                desc="Preprocessing",
            )

        # tokenize
        if "input_ids" not in dataset.column_names:
            dataset = self.map(
                dataset,
                self.tokenize,
                batched=True,
                remove_columns=dataset.column_names,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Tokenizing",
            )

        # postprocess

        if self.data_args.concatenate_data:
            # group text
            dataset = self.map(
                dataset,
                function=self.group_text,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Concatenating data",
            )

        if add_labels:
            # add labels
            dataset = self.map(
                dataset,
                function=self.add_labels,
                batched=False,  # not compatible with batching, need row lengths
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Adding labels",
            )

        if self.PROMPT_KEY in dataset:
            del dataset[self.PROMPT_KEY]

        return dataset

    def load_dataset(self):
        """
        Load the raw dataset from Hugging Face, using cached copy if available

        :param cache_dir: disk location to search for cached dataset
        :return: the requested dataset
        """
        if self.data_args.dataset_path is not None:
            if self.data_args.dvc_data_repository is not None:
                self.data_args.raw_kwargs["storage_options"] = {
                    "url": self.data_args.dvc_data_repository
                }
                self.data_args.raw_kwargs["data_files"] = self.data_args.dataset_path
            else:
                self.data_args.raw_kwargs["data_files"] = get_custom_datasets_from_path(
                    self.data_args.dataset_path,
                    self.data_args.dataset
                    if hasattr(self.data_args, "dataset")
                    else self.data_args.dataset_name,
                )

        return get_raw_dataset(
            self.data_args,
            None,
            split=self.split,
            streaming=self.data_args.streaming,
            **self.data_args.raw_kwargs,
        )

    @cached_property
    def preprocess(self) -> Union[Callable[[Any], Any], None]:
        preprocessing_func = self.data_args.preprocessing_func

        if callable(preprocessing_func):
            return preprocessing_func

        if isinstance(preprocessing_func, str):
            if ":" in preprocessing_func:
                # load func_name from "/path/to/file.py:func_name"
                return import_from_path(preprocessing_func)
            else:
                # load from the registry
                return PreprocessingFunctionRegistry.get_value_from_registry(
                    name=preprocessing_func
                )

        return self.dataset_template

    @property
    def dataset_template(self) -> Union[Callable[[Any], Any], None]:
        return None

    def tokenize(self, dataset: DatasetType) -> DatasetType:
        # manually swap text argument if specified
        if self.data_args.text_column not in dataset.column_names:
            dataset["text"] = dataset[self.data_args.text_column]

        # tokenize
        dataset = self.processor(
            **dataset,
            padding=(
                False
                if self.data_args.concatenate_data
                else "max_length"
                if self.data_args.pad_to_max_length
                else False
            ),
            max_length=self.max_seq_length,
            truncation=True,
        )

        # store unpadded prompt so we can mask out correct number of elements
        # in the labels
        if self.PROMPT_KEY in dataset:
            dataset[self.PROMPT_KEY] = self.processor(
                dataset[self.PROMPT_KEY],
                max_length=self.max_seq_length,
                truncation=True,
            )["input_ids"]

        return dataset

    def group_text(self, data):
        concatenated_data = {k: sum(data[k], []) for k in data.keys()}
        total_length = len(concatenated_data[list(data.keys())[0]])
        total_length = (total_length // self.max_seq_length) * self.max_seq_length
        result = {
            k: [
                t[i : i + self.max_seq_length]
                for i in range(0, total_length, self.max_seq_length)
            ]
            for k, t in concatenated_data.items()
        }
        return result

    def add_labels(self, data):
        # if the dataset uses prompts, mask them out so they don't contribute
        # to the loss calculation
        prompt_len = 0
        if self.PROMPT_KEY in data:
            prompt_len = len(data[self.PROMPT_KEY])
        data["labels"] = data["input_ids"].copy()
        data["labels"][:prompt_len] = [LABELS_MASK_VALUE] * prompt_len

        # mask out padding in the labels as well
        padding = len(data["attention_mask"]) - sum(data["attention_mask"])
        if padding > 0:
            data["labels"][-padding:] = [LABELS_MASK_VALUE] * padding
        return data

    def map(
        self,
        dataset: Union[Dataset, IterableDataset],
        map_fn: Union[Callable[[Any], Any], None],
        **kwargs,
    ) -> Union[Dataset, IterableDataset]:
        """
        Wrapper function around Dataset.map and IterableDataset.map, clears invalid
        parameters in the case where streaming is enabled

        :param dataset: dataset to apply mapping to
        :param kwargs: args to pass on to map function
        :return: mapped dataset
        """
        if map_fn is None:
            return dataset

        if isinstance(dataset, IterableDataset):
            # remove arguments that don't apply to streaming
            kwargs.pop("num_proc", None)
            kwargs.pop("load_from_cache_file", None)
            kwargs.pop("desc", None)

        dataset = dataset.map(map_fn, **kwargs)

        if isinstance(dataset, IterableDataset):
            dataset = dataset._resolve_features()

        return dataset
