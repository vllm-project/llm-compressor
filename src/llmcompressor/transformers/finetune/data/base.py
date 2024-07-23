from typing import Optional, Union

from compressed_tensors.registry import RegistryMixin
from datasets import Dataset, IterableDataset
from loguru import logger
from transformers import AutoTokenizer

from llmcompressor.transformers.finetune.data.data_args import DataTrainingArguments
from llmcompressor.transformers.finetune.data.data_helpers import (
    LABELS_MASK_VALUE,
    get_custom_datasets_from_path,
    get_raw_dataset,
)


class TextGenerationDataset(RegistryMixin):
    """
    Base class for text datasets, handles tokenization and dataset splits

    :param text_column: name of column corresponding to text in the dataset
    :param data_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
    :param tokenizer: tokenizer to use on dataset
    """

    PROMPT_KEY = "prompt"

    def __init__(
        self,
        text_column: str,
        data_args: DataTrainingArguments,
        split: str,
        tokenizer: AutoTokenizer,
    ):
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.raw_kwargs = data_args.raw_kwargs or {}
        self.split = split
        self.dvc_dataset = (
            True if self.data_args.dvc_data_repository is not None else False
        )
        self.custom_dataset = True if self.data_args.dataset_path is not None else False

        # configure padding
        if data_args.concatenate_data:
            self.padding = False
        elif data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            self.padding = False

        if self.tokenizer:
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # configure sequence length
        max_seq_length = data_args.max_seq_length
        model_max_length = tokenizer.model_max_length if tokenizer else max_seq_length
        if self.tokenizer and max_seq_length > model_max_length:
            logger.warning(
                f"The max_seq_length passed ({max_seq_length}) is larger than "
                f"the maximum length for the model ({tokenizer.model_max_length}). "
                f"Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, model_max_length)

    def get_raw_dataset(self, cache_dir: Optional[str] = None) -> Dataset:
        """
        Load the raw dataset from Hugging Face, using cached copy if available

        :param cache_dir: disk location to search for cached dataset
        :return: the requested dataset
        """
        if self.custom_dataset:
            if self.dvc_dataset:
                self.raw_kwargs["storage_options"] = {
                    "url": self.data_args.dvc_data_repository
                }
                self.raw_kwargs["data_files"] = self.data_args.dataset_path
            else:
                self.raw_kwargs["data_files"] = get_custom_datasets_from_path(
                    self.data_args.dataset_path,
                    self.data_args.dataset
                    if hasattr(self.data_args, "dataset")
                    else self.data_args.dataset_name,
                )

        return get_raw_dataset(
            self.data_args,
            cache_dir,
            split=self.split,
            streaming=self.data_args.streaming,
            **self.raw_kwargs,
        )

    def tokenize_and_process(
        self, raw_dataset: Optional[Dataset] = None, add_labels: Optional[bool] = True
    ) -> Dataset:
        """
        Sets up the raw dataset for finetuning, performs tokenization, concatenates
        entries to max sequence length if desired, and adds labels to each entry

        :param raw_dataset: dataset to process
        :param add_labels: whether to include labels in tokenized output
        """

        # helper fn for tokenizing text column
        def tokenize_fn(data):
            result = self.tokenizer(
                data[self.text_column],
                padding=self.padding,
                max_length=self.max_seq_length,
                truncation=True,
            )

            # store unpadded prompt so we can mask out correct number of elements
            # in the labels
            if self.PROMPT_KEY in data:
                result[self.PROMPT_KEY] = self.tokenizer(
                    data[self.PROMPT_KEY],
                    max_length=self.max_seq_length,
                    truncation=True,
                )["input_ids"]

            return result

        # helper fn for filling to max_sequence_length by concatenating entries
        def group_text_fn(data):
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

        # helper fn for adding labels, needed for loss calculation
        def label_fn(data):
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

        if raw_dataset is None:
            raw_dataset = self.get_raw_dataset()

        dataset = self.map(
            raw_dataset,
            function=tokenize_fn,
            batched=True,
            remove_columns=[self.text_column],
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        if self.data_args.concatenate_data:
            dataset = self.map(
                dataset,
                function=group_text_fn,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Grouping text",
            )

        if isinstance(dataset, IterableDataset):
            # so we can get column names from streamed_datasets
            dataset = dataset._resolve_features()

        column_names = dataset.column_names
        if isinstance(column_names, dict):
            column_names = column_names[list(column_names)[0]]

        if add_labels:
            dataset = self.map(
                dataset,
                function=label_fn,
                batched=False,  # not compatible with batching, need row lengths
                remove_columns=[self.PROMPT_KEY]
                if self.PROMPT_KEY in column_names
                else None,
                num_proc=self.data_args.preprocessing_num_workers,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Adding labels",
            )
        else:
            dataset = self.map(
                dataset,
                batched=False,  # not compatible with batching, need row lengths
                remove_columns=[self.PROMPT_KEY]
                if self.PROMPT_KEY in column_names
                else None,
            )

        return dataset

    def map(
        self, dataset: Union[Dataset, IterableDataset], **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Wrapper function around Dataset.map and IterableDataset.map, clears invalid
        parameters in the case where streaming is enabled

        :param dataset: dataset to apply mapping to
        :param kwargs: args to pass on to map function
        :return: mapped dataset
        """
        if isinstance(dataset, IterableDataset):
            # remove arguments that don't apply to streaming
            kwargs.pop("num_proc", None)
            kwargs.pop("load_from_cache_file", None)
            kwargs.pop("desc", None)

        return dataset.map(**kwargs)
