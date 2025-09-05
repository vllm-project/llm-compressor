"""
Custom dataset implementation for JSON and CSV data sources.

This module provides a CustomDataset class for loading and processing
local JSON and CSV files for text generation fine-tuning. Supports
flexible data formats and custom preprocessing pipelines for
user-provided datasets.
"""

from llmcompressor.transformers.finetune.data import TextGenerationDataset


@TextGenerationDataset.register(name="custom", alias=["json", "csv"])
class CustomDataset(TextGenerationDataset):
    """
    Child text generation class for custom local dataset supporting load
    for csv and json

    :param dataset_args: configuration settings for dataset loading
    :param split: split from dataset to load, for instance `test` or `train[:5%]`
        Can also be set to None to load all the splits
    :param processor: processor or tokenizer to use on dataset

    """

    pass
