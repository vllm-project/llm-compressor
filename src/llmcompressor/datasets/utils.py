from typing import Union

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


def get_raw_dataset(
    path: str,
    **kwargs,
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """
    Load HF alias or a dataset in local path.

    :param path: Path or name of the dataset. Accepts HF dataset stub or
        local file directory in csv, json, parquet, etc.
        If local path is provided, it must be
            1. Download path where HF dataset was downloaded to
            2. File path containing any of train, test, validation in its name
            with the supported extentions: json, jsonl, csv, arrow, parquet, text,
            and xlsx. Ex. foo-train.csv, foo-test.csv

            If a custom name is to be used, its mapping can be specified using
            `data_files` input_arg.

    :return: the requested dataset

    """
    return load_dataset(
        path,
        **kwargs,
    )
