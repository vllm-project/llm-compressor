import inspect
from functools import cached_property
from inspect import _ParameterKind as Kind
from typing import Any, Callable, Dict, List, Union

from compressed_tensors.registry import RegistryMixin
from datasets import Dataset, IterableDataset
from datasets.formatting.formatting import LazyRow
from loguru import logger

from llmcompressor.args import DatasetArguments
from llmcompressor.transformers.finetune.data.data_helpers import (
    LABELS_MASK_VALUE,
    get_custom_datasets_from_path,
    get_raw_dataset,
)
from llmcompressor.transformers.utils.preprocessing_functions import (
    PreprocessingFunctionRegistry,
)
from llmcompressor.typing import DatasetType, Processor
from llmcompressor.utils import import_from_path


class TextGenerationDataset(RegistryMixin):
    """
    Base class for text datasets. Applies the following transformations to a dataset
    in order to prepare the dataset to be loaded by a dataloader

    1. Load dataset from huggingface or local cache
    2. Preprocess dataset according to preprocess function or chat/dataset template
    3. Tokenize dataset using model tokenizer/processor
    4. Apply post processing such as grouping text and/or adding labels for finetuning

    :param dataset_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param processor: processor or tokenizer to use on dataset
    """

    # used to mask out the prompt so prompt tokens do not contribute to training loss
    PROMPT_KEY = "prompt"

    def __init__(
        self,
        dataset_args: DatasetArguments,
        split: str,
        processor: Processor,
    ):
        self.dataset_args = dataset_args
        self.split = split
        self.processor = processor

        # get tokenizer
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

        if self.tokenizer is not None:
            # fill in pad token
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # configure sequence length
            max_seq_length = dataset_args.max_seq_length
            if dataset_args.max_seq_length > self.tokenizer.model_max_length:
                logger.warning(
                    f"The max_seq_length passed ({max_seq_length}) is larger than "
                    f"maximum length for model ({self.tokenizer.model_max_length}). "
                    f"Using max_seq_length={self.tokenizer.model_max_length}."
                )
            self.max_seq_length = min(
                dataset_args.max_seq_length, self.tokenizer.model_max_length
            )

            # configure padding
            self.padding = (
                False
                if self.dataset_args.concatenate_data
                else "max_length"
                if self.dataset_args.pad_to_max_length
                else False
            )

        else:
            self.max_seq_length = None
            self.padding = False

    def __call__(self, add_labels: bool = True) -> DatasetType:
        dataset = self.dataset_args.dataset

        if isinstance(dataset, str):
            # load dataset: load from huggingface or disk
            dataset = self.load_dataset()
        logger.debug(f"Raw dataset: {get_columns(dataset)}")

        if self.preprocess is not None:
            # preprocess: apply template or preprocessing function
            dataset = self.map(
                dataset,
                self.preprocess,
                batched=False,
                num_proc=self.dataset_args.preprocessing_num_workers,
                load_from_cache_file=not self.dataset_args.overwrite_cache,
                desc="Preprocessing",
            )
            logger.debug(f"Dataset after preprocessing: {get_columns(dataset)}")

        # rename and remove columns match processor kwargs
        dataset = self.rename_columns(dataset)
        logger.debug(f"Dataset after column renaming: {get_columns(dataset)}")

        # use processor.model_input_names to determine if the ds is already tokenized
        model_input_names = getattr(self.processor, "model_input_names", ["input_ids"])
        if not any(col_name in model_input_names for col_name in get_columns(dataset)):
            # tokenize/ process
            dataset = self.filter_tokenizer_args(dataset)
            logger.debug(f"Tokenizer args after filtering: {get_columns(dataset)}")
            dataset = self.map(
                dataset,
                self.tokenize,
                batched=False,  # batching is not well supported for vision processors
                keep_in_memory=True,  # bug occurs when not batched and not in memory,
                # subsequent ds.map calls are always batched,
                # regardless of `batched` argument
                remove_columns=get_columns(dataset),  # assumes that input names
                # and output names are disjoint
                num_proc=self.dataset_args.preprocessing_num_workers,
                load_from_cache_file=not self.dataset_args.overwrite_cache,
                desc="Tokenizing",
            )
            logger.debug(f"Model kwargs after tokenizing: {get_columns(dataset)}")

        if self.dataset_args.concatenate_data:
            # postprocess: group text
            dataset = self.map(
                dataset,
                self.group_text,
                batched=True,
                num_proc=self.dataset_args.preprocessing_num_workers,
                load_from_cache_file=not self.dataset_args.overwrite_cache,
                desc="Concatenating data",
            )
            logger.debug(f"Model kwargs after concatenating: {get_columns(dataset)}")

        if add_labels:
            # postprocess: add labels
            dataset = self.map(
                dataset,
                self.add_labels,
                batched=False,  # not compatible with batching, need row lengths
                num_proc=self.dataset_args.preprocessing_num_workers,
                load_from_cache_file=not self.dataset_args.overwrite_cache,
                desc="Adding labels",
            )
            logger.debug(f"Model kwargs after adding labels: {get_columns(dataset)}")

        elif self.PROMPT_KEY in get_columns(dataset):
            dataset = dataset.remove_columns(self.PROMPT_KEY)
            logger.debug("Removed prompt key")

        logger.debug(f"Model kwargs after postprocessing: {get_columns(dataset)}")
        return dataset

    def load_dataset(self):
        """
        Load the raw dataset from Hugging Face, using cached copy if available

        :param cache_dir: disk location to search for cached dataset
        :return: the requested dataset
        """
        if self.dataset_args.dataset_path is not None:
            if self.dataset_args.dvc_data_repository is not None:
                self.dataset_args.raw_kwargs["storage_options"] = {
                    "url": self.dataset_args.dvc_data_repository
                }
                self.dataset_args.raw_kwargs["data_files"] = (
                    self.dataset_args.dataset_path
                )
            else:
                self.dataset_args.raw_kwargs["data_files"] = (
                    get_custom_datasets_from_path(
                        self.dataset_args.dataset_path,
                        self.dataset_args.dataset
                        if hasattr(self.dataset_args, "dataset")
                        else self.dataset_args.dataset_name,
                    )
                )

        logger.debug(f"Loading dataset {self.dataset_args.dataset}")
        return get_raw_dataset(
            self.dataset_args,
            None,
            split=self.split,
            streaming=self.dataset_args.streaming,
            **self.dataset_args.raw_kwargs,
        )

    @cached_property
    def preprocess(self) -> Union[Callable[[LazyRow], Any], None]:
        """
        The function must return keys which correspond to processor/tokenizer kwargs,
        optionally including PROMPT_KEY
        """
        preprocessing_func = self.dataset_args.preprocessing_func

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

    def rename_columns(self, dataset: DatasetType) -> DatasetType:
        # rename columns to match processor/tokenizer kwargs
        column_names = get_columns(dataset)
        if self.dataset_args.text_column in column_names and "text" not in column_names:
            logger.debug(f"Renaming column `{self.dataset_args.text_column}` to `text`")
            dataset = dataset.rename_column(self.dataset_args.text_column, "text")

        return dataset

    def filter_tokenizer_args(self, dataset: DatasetType) -> DatasetType:
        # assumes that inputs are not passed via self.processor.__call__ args and kwargs
        signature = inspect.signature(self.processor.__call__)
        tokenizer_args = set(
            key
            for key, param in signature.parameters.items()
            if param.kind not in (Kind.VAR_POSITIONAL, Kind.VAR_KEYWORD)
        )
        logger.debug(
            f"Found processor args `{tokenizer_args}`. Removing all other columns"
        )

        column_names = get_columns(dataset)
        return dataset.remove_columns(
            list(set(column_names) - set(tokenizer_args) - set([self.PROMPT_KEY]))
        )

    def tokenize(self, data: LazyRow) -> Dict[str, Any]:
        # separate prompt
        prompt = data.pop(self.PROMPT_KEY, None)

        # tokenize
        data = self.processor(
            **data,
            padding=self.padding,
            max_length=self.max_seq_length,
            truncation=True,
        )

        # store unpadded prompt so we can mask out correct number of elements in labels
        if prompt is not None:
            data[self.PROMPT_KEY] = self.processor(
                text=prompt,
                max_length=self.max_seq_length,
                truncation=True,
            )["input_ids"]

        return data

    def group_text(self, data: LazyRow) -> Dict[str, Any]:
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

    def add_labels(self, data: LazyRow) -> LazyRow:
        if "pixel_values" in data:
            raise NotImplementedError(
                "Label masking for vision datasets has not been implemented yet"
            )

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
        function: Callable[[Any], Any],
        **kwargs,
    ) -> Union[Dataset, IterableDataset]:
        """
        Wrapper function around Dataset.map and IterableDataset.map.

        If the dataset is streaming (in the case of IterableDataset), non-applicable
        arguments are ignored and the dataset features are resolved
        """
        if isinstance(dataset, IterableDataset):
            # remove arguments that don't apply to streaming
            kwargs.pop("num_proc", None)
            kwargs.pop("load_from_cache_file", None)
            kwargs.pop("desc", None)
            kwargs.pop("keep_in_memory", None)

        dataset = dataset.map(function, **kwargs)

        if isinstance(dataset, IterableDataset):
            dataset = dataset._resolve_features()

        return dataset


def get_columns(dataset: DatasetType) -> List[str]:
    column_names = dataset.column_names
    if isinstance(column_names, dict):
        column_names = sum(column_names.values(), [])

    return column_names
